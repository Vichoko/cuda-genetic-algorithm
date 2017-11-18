#include <stdio.h>
#include <stdlib.h>

#include <limits.h>

#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#include "ga.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

// MACROS
#define NUM_BLOCKS 216
// GEFORCE GTX 960m has 640 blocks.
#define THREADS_PER_BLOCK 4
// Island is a block. Every individual is a thread.
#define TOTAL_POPULATION NUM_BLOCKS*THREADS_PER_BLOCK
// azucar sintactico
#define ISLAND_POPULATION THREADS_PER_BLOCK
// azucar sintactico

#define MIGRATION_CHANCE 0.25
//chance of migration ocurrs
#define MIGRATION_SIZE 2
// Quantity of the island individuals will migrate
#define MUTATION_CHANCE 0.1
//chance of ocurring a mutation in genes
#define CROSSOVER_CHANCE 0.5

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

__host__ unsigned char* file_to_byte_array(const char* filename) { // TODO: add to init_resources
	printf("loading img\n");

	FILE *fileptr;
	unsigned char *buffer;

	fileptr = fopen(filename, "rb");  // Open the file in binary mode
	fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
	genes_len = (unsigned int) ftell(fileptr) + 1; // Get the current byte offset in the file  + \0

	printf("genes size (bytes): %d\n", genes_len);
	rewind(fileptr);                   // Jump back to the beginning of the file

	buffer = (unsigned char*) malloc(genes_len * sizeof(unsigned char)); // Enough memory for file
	fread(buffer, genes_len, 1, fileptr); // Read in the entire file
	fclose(fileptr); // Close the file
	printf("	done loading img\n");

	return buffer;
}

__device__ int fitness_fun(unsigned char* actual_genes,
		unsigned char* expected_genes, int genes_len) {
	if (actual_genes == NULL) {
		printf("actual genes are NULL\n");

	}
	//unsigned char* expected_genes; // TODO: this should be the target BMP image.
	int match_counter = 0;

	for (int geneIndex = 0; geneIndex < genes_len; geneIndex++) {
		unsigned char a_byte = expected_genes[geneIndex];
		unsigned char b_byte = actual_genes[geneIndex];
		unsigned char res_byte = ~(a_byte ^ b_byte);

		// now count bits in 1
		while (res_byte > 0) {
			if ((res_byte & 1) == 1) {
				match_counter++;
			}
			res_byte >>= 1;
		}
	}
	return match_counter;
}

// framework privates
//Device
__device__ void migrate_best_genes(int localIndex, const unsigned int genes_len,
		int* migrate_flag, int* s_fitness, unsigned char* g_all_genes,
		unsigned char* s_genes, unsigned char* g_expected_genes) {
	/** If migrate, the top MIGRATION_SIZE individuals goes to other island,
	 * and worst MIGRATION_SIZE individuals get replaced by other island's */
	if (*migrate_flag) {
		// do migration

		// this is a very unneficient way to sort, todo: optimize sorting
		int sorted_position = ISLAND_POPULATION - 1;
		int sorted_offset = 0; // in case of repeated values, needed to avoid repetition of index
		for (int brotherIndex = 0; brotherIndex < ISLAND_POPULATION;
				brotherIndex++) {
			if (brotherIndex == localIndex) {
				continue;
			}
			if (s_fitness[brotherIndex] < s_fitness[localIndex]) { //find position in array
				sorted_position--;
			} else if (s_fitness[brotherIndex] == s_fitness[localIndex]
					&& localIndex < brotherIndex) {
				sorted_offset++;
			}
		}
		sorted_position -= sorted_offset;
		//sorted_position is the index where this gene should be located if population is sorted descending

		/*** BEGIN CRITICAL SECTION; evaluate if use mutex ***/
		/** MIGRATION_SIZE threads copy best elements to Global Memory **/
		if (sorted_position < MIGRATION_SIZE) { // 0,1,2,...,MIGRATION_SIZE-1 best elements
			// Copy MIGRATION_SIZE best elements from shared to global memory
			int island_genes_len = genes_len * ISLAND_POPULATION;
			memcpy(
					&g_all_genes[blockIdx.x * island_genes_len
							+ sorted_position * genes_len], // sorted position in global memory
					&s_genes[localIndex * genes_len], // this genes are one of the best
					sizeof(unsigned char) * genes_len);

		}
		/** MIGRATION_SIZE threads replaces worst elements from Global Memory (other island)*/
		else if (sorted_position >= (int) ISLAND_POPULATION - MIGRATION_SIZE) { // ISLAND_POPULATION-1, ..., ISLAND_POPULATION-MIGRATION_SIZE worst elements
			// Replace MIGRATION_SIZE worst elements from global to shared memory with best neighbor values
			int island_genes_len = genes_len * ISLAND_POPULATION;
			int max_index = ISLAND_POPULATION - 1 - sorted_position; // from 0 to MIG_SIZE - 1
			unsigned char* genes = &s_genes[localIndex * genes_len];

			//Circular chain of blocks, if block is last, neighbor is the first one.
			memcpy(
					genes, // this genes are one of the worst
					&g_all_genes[(
							blockIdx.x >= NUM_BLOCKS - 1 ? 0 : (blockIdx.x + 1))
							* island_genes_len + max_index * genes_len], // get best genes from neighbor block
					sizeof(unsigned char) * genes_len);

			// update fitness in local
			s_fitness[localIndex] = fitness_fun(genes, g_expected_genes,
					genes_len);
		}
		/*** END CRITICAL SECTION; evaluate if using mutex ***/
		__syncthreads(); // wait every thread to finish, then restore migrate_flag
		*migrate_flag = 0;
	}
}

__device__ int tournament_selection(int k, curandState_t state,
		int* s_fitness) {
	int max_fitness = -1;
	int max_individual = -1;

	for (int i = 0; i < k; i++) {
		int random = curand(&state) % ISLAND_POPULATION; // get random individidual
		if (s_fitness[random] > max_fitness) {
			max_fitness = s_fitness[random];
			max_individual = random;
		}
	}
	return max_individual;
}

__device__ void crossover(int dad, int mom, unsigned char* son_genes,
		unsigned char* s_genes, unsigned int genes_len, curandState_t state) {
	unsigned int random = curand(&state); //32 bits of pure randomness
	unsigned int random2 = curand(&state); //32 bits of pure randomness
	unsigned int random3 = curand(&state); //32 bits of pure randomness
	unsigned int mom_or_dad = random2 % 2;

	if (!(random != random2 || random != random3)) {
		printf("nod ??????? dif\n");
	} // todo: delete

	if (random < CROSSOVER_CHANCE * UINT_MAX) {
		// do two-point cross over
		//todo: decide 2 points of genes
		unsigned int p1 = random2 % genes_len; // first cell of crossed over genes
		unsigned int p2 = random3 % genes_len; // last cell of crossed over genes

		if (p2 < p1) { // make p1 the smaller index
			unsigned int aux_p = p1;
			p1 = p2;
			p2 = aux_p;
		}
		p2++; // p2 is first cell of third segment

		memcpy(son_genes,
				&s_genes[(mom_or_dad ? mom : dad) * genes_len],
				sizeof(unsigned char) * p1); // first segment, 0 to p1-1, inclusive

		memcpy(son_genes,
				&s_genes[(mom_or_dad ? dad : mom) * genes_len + p1],
				sizeof(unsigned char) * (p2 - p1)); // second, p1 to p2-1, inc

		memcpy(son_genes,
				&s_genes[(mom_or_dad ? mom : dad) * genes_len + p2],
				sizeof(unsigned char) * (genes_len- p2)); // third, p2 to genes_len-1

	} else {
		// be equal as dad or mom
		memcpy(son_genes, &s_genes[(mom_or_dad ? mom : dad) * genes_len],
				sizeof(unsigned char) * genes_len);
	}
}

__device__ void mutate(unsigned char* son_genes, unsigned int genes_len, curandState_t state){
	// each bit has p = 1/genes_len of switch its value
	//make mask with bbits that shoud be switched
	float bit_switch_probability = 1.0f/ (genes_len* 8);

	for (int geneIndex= 0; geneIndex < genes_len; geneIndex++){
		unsigned char switch_mask = 0;
		for (int i= 0; i< 8; i++){

			unsigned int random = curand(&state); //32 bits of pure randomness
			if (random<= bit_switch_probability* UINT_MAX){
				switch_mask |= 1;
			}
			switch_mask <<= 1; //1 if switch, 0 if dont
		}
		son_genes[geneIndex] ^= switch_mask;
	}
}

//Kernels
__global__ void island_controller(const unsigned int genes_len,
		unsigned char* g_all_genes, unsigned char* g_best_genes,
		unsigned int* g_finish_signal, unsigned int seed, unsigned int verbose,
		unsigned char* g_expected_genes //custom param, could be in shared memory because it is used too much
		) {
	extern __shared__ int shared_array[];
	int* s_fitness = shared_array;
	unsigned char* s_genes = (unsigned char*) &s_fitness[ISLAND_POPULATION];
	unsigned char* s_son_genes = (unsigned char*) &s_genes[ISLAND_POPULATION
			* genes_len];
	int* migrate_flag = (int*) &s_son_genes[ISLAND_POPULATION * genes_len];

	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; // indice del individuo/thread
	int localIndex = threadIdx.x;

	/* CUDA's random number library */
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
	(globalIndex + 1), /* the sequence number is only important with multiple cores */
	genes_len, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	&state);

	/* copy genes from Global Memory (generate_initial_population) to Shared */
	if (localIndex == 0) {
		// one thread copy all the island genes to shared memory
		int island_genes_len = genes_len * ISLAND_POPULATION;
		memcpy(s_genes, &g_all_genes[blockIdx.x * island_genes_len],
				sizeof(unsigned char) * island_genes_len);
		//initial conditions
		*migrate_flag = 0;
	}
	__syncthreads();

	/** Evolution loop */
	unsigned int last_gen = 100;

	while (*g_finish_signal == 0 && last_gen--) { // todo: while cuando se cambie la flag
		/** Update fitness in Shared memory */
		unsigned char* genes = &s_genes[localIndex * genes_len];
		s_fitness[localIndex] = fitness_fun(genes, g_expected_genes, genes_len);
		__syncthreads();

		/** Migrate? Yes or No */
		if (localIndex == 0) {
			// calculate migration condition
			unsigned int random = curand(&state);
			if (random < MIGRATION_CHANCE * UINT_MAX) {
				*migrate_flag = 1;
			}
		}
		__syncthreads(); // all threads wait for the migrate_flag update

		/** Wit a chance, migrate the top MIGRATION_SIZE individuals to other island,
		 * and worst MIGRATION_SIZE individuals get replaced by other island's best */
		migrate_best_genes(localIndex, genes_len, migrate_flag, s_fitness,
				g_all_genes, s_genes, g_expected_genes);
		__syncthreads();

		// select two parents for new individual, this thread is in charge of that son
		int dad = tournament_selection(3, state, s_fitness);
		int mom = tournament_selection(3, state, s_fitness);
		__syncthreads();

		unsigned char* son_genes = &s_son_genes[localIndex * genes_len];
		crossover(dad, mom, son_genes, s_genes, genes_len, state);
		//son_genes contain crossed over genes of dad and mom
		__syncthreads();

		mutate(son_genes, genes_len, state);
		__syncthreads();

		memcpy(&s_genes[localIndex * genes_len], son_genes,
				sizeof(unsigned char) * genes_len);
		s_fitness[localIndex] = fitness_fun(son_genes, g_expected_genes,
				genes_len); //actualizar genes y fitness en shared memory

		/** BEGIN CRITICAL SECTION **/
		if (s_fitness[localIndex] >= genes_len * 8 && *g_finish_signal == 0) { // fitness is number of common bits
			*g_finish_signal = 1;
			memcpy(g_best_genes, son_genes, sizeof(unsigned char) * genes_len);
			break;
		}
		/** END CRITICAL SECTION **/
	}
}



__global__ void generate_initial_population(const unsigned int genes_len,
		unsigned char* all_genes, unsigned int seed) {
	// cada thread se encarga de generarse a si mismo y colocarse en memoria principal de GPU.
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; // indice del individuo/thread
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	 we will store a random state for every thread  */
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
	(globalIndex + 1), /* the sequence number is only important with multiple cores */
	genes_len, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	&state);

	for (int geneIndex = 0; geneIndex < genes_len; geneIndex++) {
		/* we have to initialize the state */

		// se debe poder optimizar
		/* curand works like rand - except that it takes a state as a parameter */
		unsigned int result = curand(&state);
		all_genes[geneIndex + globalIndex * genes_len] = (unsigned char) result
				% 256; // write in global device memory
	}
}

//Host
__host__ void init_resources() {
	printf("Initializing resources.\n");
	// custom case, image
	unsigned char* h_expected_genes = file_to_byte_array(
			"/home/vichoko/eclipse-workspace/cuda-GA/image.bmp");

	// debug
	//FILE* file = fopen( "exported_img.bmp", "wb" );
	//fwrite(h_expected_genes, 1, genes_len, file);

	gpuErrchk(cudaMalloc(&d_expected_genes, sizeof(unsigned char) * genes_len)); // arreglo de genes aplanado inicialmente vacio
	gpuErrchk(
			cudaMemcpy(d_expected_genes, h_expected_genes,
					genes_len * sizeof(unsigned char), cudaMemcpyHostToDevice)); // arreglo de genes aplanado inicialmente vacio
	gpuErrchk(cudaDeviceSynchronize());

	// general purpose
	gpuErrchk(
			cudaMalloc(&d_all_genes, sizeof(unsigned char)*genes_len*TOTAL_POPULATION)); // arreglo de genes aplanado inicialmente vacio
	gpuErrchk(cudaMalloc(&d_best_genes, sizeof(unsigned char) * genes_len));
	gpuErrchk(cudaMalloc(&d_finish_signal, sizeof(int)));
	printf("	done initializing resources.\n");
}

__host__ void init_population() {
	printf("Generating initial population.\n");
	generate_initial_population<<<
	NUM_BLOCKS,
	THREADS_PER_BLOCK, sizeof(unsigned char) * genes_len * ISLAND_POPULATION // tamaño de memoria compartida
	>>>(genes_len, d_all_genes, time(NULL));

	gpuErrchk(cudaPeekAtLastError());

	/**
	 unsigned char* h_all_genes = (unsigned char*) malloc(sizeof(unsigned char)* genes_len* TOTAL_POPULATION);
	 gpuErrchk( cudaMemcpy(
	 h_all_genes,
	 d_all_genes,
	 sizeof(unsigned char)* genes_len* TOTAL_POPULATION,
	 cudaMemcpyDeviceToHost));

	 */ // debug, todo: delete
	gpuErrchk(cudaDeviceSynchronize());

	printf("	done Generating initial population.\n");
}
__host__ unsigned char* run_evolution() {
	// calculate size of shared memory
	/** need to store island genes and fitnesses **/
	int shared_mem_size = sizeof(unsigned char) * genes_len * ISLAND_POPULATION; // for parent genes
	shared_mem_size += sizeof(unsigned char) * genes_len * ISLAND_POPULATION; // for son genes
	shared_mem_size += sizeof(int) * ISLAND_POPULATION; // for fitnesses
	shared_mem_size += sizeof(int); // for migration flag

	unsigned int verbose = 1;
	printf("Starting evolution\n");
	island_controller<<<
	NUM_BLOCKS, 	// NUMBER OF ISLANDS
			THREADS_PER_BLOCK, // ISLAND_POPULATION
			shared_mem_size // tamaño de memoria compartida
	>>>(genes_len, d_all_genes, d_best_genes, d_finish_signal, time(NULL), // random seed
	verbose, //todo: delete
			d_expected_genes); //custom param

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaThreadSynchronize() );
	//gpuErrchk(cudaDeviceSynchronize());
	// if im here, perfect individual should exist

	unsigned char* best_genes = (unsigned char*) malloc(
			sizeof(unsigned char) * genes_len);
	gpuErrchk(
			cudaMemcpy(best_genes, d_best_genes,
					sizeof(unsigned char) * genes_len, cudaMemcpyDeviceToHost));

	printf("	Evolution finished.\n");
	return best_genes; // Todo: asegurar que sea el individuo perfecto
}

// CPU Controller
int main() {
	printf("Started CUDA GA.\n");
	init_resources();
	init_population();

	unsigned char* evoluted_genes = run_evolution();
	FILE* file = fopen("evoluted_img.bmp", "wb");
	fwrite(evoluted_genes, 1, genes_len, file);
	return 0;
}

