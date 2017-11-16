#include <stdio.h>
#include <stdlib.h>

#include <limits.h>

#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#include "ga.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// MACROS
#define NUM_BLOCKS 512
// GEFORCE GTX 960m has 640 blocks.
#define THREADS_PER_BLOCK 20
// Island is a block. Every individual is a thread.
#define TOTAL_POPULATION NUM_BLOCKS*THREADS_PER_BLOCK
// azucar sintactico
#define ISLAND_POPULATION THREADS_PER_BLOCK
// azucar sintactico

#define MIGRATION_CHANCE 0.25 //chance of migration ocurrs
#define MIGRATION_SIZE 5 // Quantity of the island individuals will migrate
#define MUTATION_CHANCE 0.1 //chance of ocurring a mutation in genes



// GLOBALS
//device
unsigned int* d_finish_signal;
unsigned char* d_all_genes;
unsigned char* d_best_genes;
//host
unsigned char* best_genes;

// Esquema de framework implica modificar funcion de fitness y tamaño de gen, para adaptarlo al problema.

// example globals
//host
unsigned int genes_len; // TODO: size should be image size in bytes. ¿que pasa si imagen no termian en byte completo?

//device
unsigned char* d_expected_genes;

__host__ unsigned char* file_to_byte_array(const char* filename){ // TODO: add to init_resources
	printf("loading img\n");

	FILE *fileptr;
	unsigned char *buffer;

	fileptr = fopen(filename, "rb");  // Open the file in binary mode
	fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
	genes_len = (unsigned int) ftell(fileptr)+1;             // Get the current byte offset in the file  + \0

	printf("genes size (bytes): %d\n", genes_len);
	rewind(fileptr);                      // Jump back to the beginning of the file

	buffer = (unsigned char*) malloc(genes_len*sizeof(unsigned char)); // Enough memory for file
	fread(buffer, genes_len, 1, fileptr); // Read in the entire file
	fclose(fileptr); // Close the file
	printf("	done loading img\n");

	return buffer;
}

__device__ int fitness_fun(
		unsigned char* actual_genes,
		unsigned char* expected_genes,
		int genes_len)
{
	if (actual_genes == NULL){
		printf("actual genes are NULL\n");

	}
	//unsigned char* expected_genes; // TODO: this should be the target BMP image.
	int match_counter = 0;

	for (int geneIndex = 0; geneIndex < genes_len; geneIndex++){
		unsigned char a_byte = expected_genes[geneIndex];
		unsigned char b_byte = actual_genes[geneIndex];
		unsigned char res_byte = ~(a_byte ^ b_byte);

		// now count bits in 1
		while (res_byte > 0){
			if ((res_byte & 1) == 1){
				match_counter++;
			}
			res_byte >>= 1;
		}
	}
	return match_counter;
}


// Kernels
__global__ void island_controller(
		const unsigned int genes_len,
		unsigned char* g_all_genes,
		unsigned char* g_best_genes,
		unsigned int* g_finish_signal,
		unsigned int seed,
		unsigned int verbose,
		unsigned char* expected_genes //custom param
	)
{
	extern __shared__ int shared_array[];
	int* s_fitness = shared_array;
	unsigned char* s_genes = (unsigned char*) &s_fitness[ISLAND_POPULATION];
	int* migrate_flag = (int*) &s_genes[ISLAND_POPULATION];

	int globalIndex = blockIdx.x* blockDim.x+ threadIdx.x; // indice del individuo/thread
	int localIndex = threadIdx.x;

	/* CUDA's random number library uses curandState_t to keep track of the seed value
	 we will store a random state for every thread  */
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
			  (globalIndex+ 1), /* the sequence number is only important with multiple cores */
			genes_len, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			  &state);

	if (localIndex == 0){
		// one thread copy all the island genes to shared memory
		int island_genes_len = genes_len* ISLAND_POPULATION;
//		printf("_memcpy %d\n", blockIdx.x);
	    memcpy(
	    		s_genes,
	    		&g_all_genes[blockIdx.x* island_genes_len],
	    		sizeof(unsigned char)* island_genes_len);
	    *migrate_flag = 0;
	}
	__syncthreads();
	//printf("_evo_loop\n");
	if (*g_finish_signal == 0){ // todo: while cuando se cambie la flag
		unsigned char* genes = &s_genes[localIndex* genes_len];
		s_fitness[localIndex] = fitness_fun(genes, expected_genes, genes_len);

		//printf("_fitness %d\n", s_fitness[localIndex]);
		__syncthreads();

		if (localIndex == 0){
			// calculate migration condition
			unsigned int random = curand(&state);
			if (random < MIGRATION_CHANCE * UINT_MAX){
				*migrate_flag = 1;
			}
		}
		__syncthreads(); // all threads wait for the migrate_flag update

		if (*migrate_flag){
			// do migration
			//todo: sort fitnesses, order genes in global memory descending

			// this is a very unneficient way, todo: optimize sorting
			int sorted_position = ISLAND_POPULATION - 1;
			int sorted_offset = 0; // in case of repeated values, needed to avoid repetition of index
			for (int brotherIndex = 0; brotherIndex < ISLAND_POPULATION; brotherIndex++){
				if (brotherIndex == localIndex){
					continue;
				}
				if (s_fitness[brotherIndex]< s_fitness[localIndex]){ //find position in array
					sorted_position--;
				} else if (s_fitness[brotherIndex]== s_fitness[localIndex] && localIndex< brotherIndex){
					sorted_offset++;
				}
			}
			sorted_position -= sorted_offset;

			//sorted_position is the index where this gene should be located if population is sorted descending

			if (sorted_position< MIGRATION_SIZE){ // 0,1,2,...,MIGRATION_SIZE-1 best elements
				// Copy MIGRATION_SIZE best elements from shared to global memory
				int island_genes_len = genes_len* ISLAND_POPULATION;
				memcpy(
						&g_all_genes[blockIdx.x* island_genes_len + sorted_position* genes_len], // sorted position in global memory
						&s_genes[localIndex* genes_len], // this genes are one of the best
						sizeof(unsigned char)* genes_len);

			} else if (sorted_position>= (int) ISLAND_POPULATION - MIGRATION_SIZE){ // ISLAND_POPULATION-1, ..., ISLAND_POPULATION-MIGRATION_SIZE worst elements
				// Replace MIGRATION_SIZE worst elements from global to shared memory with best neighbor values
				int island_genes_len = genes_len* ISLAND_POPULATION;
				int max_index = ISLAND_POPULATION-1- sorted_position; // from 0 to MIG_SIZE - 1
				unsigned char* genes = &s_genes[localIndex* genes_len];

				if (blockIdx.x>= NUM_BLOCKS - 1){
					//border case, circular array of blocks
					memcpy(
							genes, // this genes are one of the worst
							&g_all_genes[max_index* genes_len], // get best genes from neighbor block
							sizeof(unsigned char)* genes_len);
				} else {
					memcpy(
							genes, // this genes are one of the worst
							&g_all_genes[(blockIdx.x + 1)* island_genes_len + max_index* genes_len], // get best genes from neigbhor block
							sizeof(unsigned char)* genes_len);
				}
				// update fitness in local
				s_fitness[localIndex] = fitness_fun(genes, expected_genes, genes_len);
			}
			__syncthreads(); // wait every thread to finish, then restore migrate_flag
			*migrate_flag = 0;
		}

		// TODO: Evaluar migracion
		__syncthreads();
		// TODO: Seleccion
		__syncthreads();
		//TODO : crossover
		__syncthreads();
		//TODO: mutacion
		__syncthreads();
		//TODO: EValuar si seguir; si no retornar genes
	}
}

__global__ void generate_initial_population(
		const unsigned int genes_len,
		unsigned char* all_genes,
		unsigned int seed)
{
	// cada thread se encarga de generarse a si mismo y colocarse en memoria principal de GPU.
	int globalIndex = blockIdx.x* blockDim.x+ threadIdx.x; // indice del individuo/thread
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	     we will store a random state for every thread  */
	  curandState_t state;
	  curand_init(seed, /* the seed controls the sequence of random values that are produced */
			  	  (globalIndex+ 1), /* the sequence number is only important with multiple cores */
			  	genes_len, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	              &state);

	for (int geneIndex= 0; geneIndex< genes_len; geneIndex++){
		  /* we have to initialize the state */

		  // se debe poder optimizar
		  /* curand works like rand - except that it takes a state as a parameter */
		  unsigned int result = curand(&state);
		  all_genes[geneIndex+ globalIndex*genes_len] = (unsigned char) result% 256; // write in global device memory
	}
}

// Host fun
__host__ void init_resources(){
	printf("Initializing resources.\n");
	// custom case, image
	unsigned char* h_expected_genes = file_to_byte_array("/home/vichoko/eclipse-workspace/cuda-GA/image.bmp");

	// debug
	//FILE* file = fopen( "exported_img.bmp", "wb" );
	//fwrite(h_expected_genes, 1, genes_len, file);

	gpuErrchk( cudaMalloc(&d_expected_genes, sizeof(unsigned char)*genes_len)); // arreglo de genes aplanado inicialmente vacio
	gpuErrchk( cudaMemcpy(d_expected_genes, h_expected_genes, genes_len*sizeof(unsigned char), cudaMemcpyHostToDevice)); // arreglo de genes aplanado inicialmente vacio
	gpuErrchk( cudaDeviceSynchronize() );


	// general purpose
	gpuErrchk( cudaMalloc(&d_all_genes, sizeof(unsigned char)*genes_len*TOTAL_POPULATION) ); // arreglo de genes aplanado inicialmente vacio
	gpuErrchk( cudaMalloc(&d_best_genes, sizeof(unsigned char)*genes_len));
	gpuErrchk( cudaMalloc(&d_finish_signal, sizeof(int)));
	printf("	done initializing resources.\n");
}

__host__ void init_population() {
	printf("Generating initial population.\n");
	generate_initial_population<<<
			NUM_BLOCKS,
			THREADS_PER_BLOCK,
			sizeof(unsigned char) * genes_len * ISLAND_POPULATION // tamaño de memoria compartida
	>>>(genes_len, d_all_genes, time(NULL));

	gpuErrchk( cudaPeekAtLastError() );

	/**
	unsigned char* h_all_genes = (unsigned char*) malloc(sizeof(unsigned char)* genes_len* TOTAL_POPULATION);
	gpuErrchk( cudaMemcpy(
			h_all_genes,
			d_all_genes,
			sizeof(unsigned char)* genes_len* TOTAL_POPULATION,
			cudaMemcpyDeviceToHost));

*/ // debug
	gpuErrchk( cudaDeviceSynchronize() );

	printf("	done Generating initial population.\n");
}
__host__ unsigned char* run_evolution() {
	// calculate size of shared memory
	/** need to store island genes and fitnesses **/
	int shared_mem_size = sizeof(unsigned char) * genes_len * ISLAND_POPULATION;
	shared_mem_size += sizeof(int) * ISLAND_POPULATION;
	shared_mem_size += sizeof(int);

	unsigned int verbose = 1;
	printf("Starting evolution\n");
	island_controller<<<
			NUM_BLOCKS,
			THREADS_PER_BLOCK,
			shared_mem_size // tamaño de memoria compartida
	>>>(
			genes_len,
			d_all_genes,
			d_best_genes,
			d_finish_signal,
			time(NULL),
			verbose,
			d_expected_genes);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	printf("	Evolution finished.\n");
	return NULL; // Todo: asegurar que sea el individuo perfecto
}

// CPU Controller
int main(){
	printf("Started CUDA GA.\n");
	init_resources();
	init_population();

	unsigned char* evoluted_genes = run_evolution();
	return 0;
}


