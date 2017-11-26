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
#define NUM_BLOCKS 1024
// GEFORCE GTX 960m has 640 blocks.
#define THREADS_PER_BLOCK 512
// Island is a block. Every individual is a thread.
#define TOTAL_POPULATION NUM_BLOCKS*THREADS_PER_BLOCK
// azucar sintactico
#define ISLAND_POPULATION THREADS_PER_BLOCK
// azucar sintactico

#define MIGRATION_CHANCE 0.3
//chance of migration ocurrs
#define MIGRATION_SIZE 128
// Quantity of the island individuals will migrate
#define MUTATION_CHANCE 1
//chance of ocurring a mutation in genes
#define CROSSOVER_CHANCE 1
#define TOURNAMENT_K 2
#define GENERATION_PER_ROUND 1000

// GLOBALS
//device
unsigned int* d_finish_signal;
unsigned char* d_all_genes;
unsigned char* d_best_genes;
//host
unsigned char* best_genes;
unsigned char* h_expected_genes;


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
		return 1;
	}
	if (expected_genes == NULL) {
		printf("expected_genes are NULL\n");
		return 2;
	}
	//unsigned char* expected_genes; // TODO: this should be the target BMP image.
	int match_counter = 0;
	for (int geneIndex = 0; geneIndex < genes_len; geneIndex++) {
		unsigned char a_byte = expected_genes[geneIndex];
		unsigned char b_byte = actual_genes[geneIndex];
		unsigned char res_byte = ~(a_byte ^ b_byte); // ^ is xor, 1 if both bits are different. That negated, are the common bits

		// now count bits in 1
		unsigned int res_int = (unsigned int) res_byte;
		match_counter += __popc(res_int); // count bits in 1 in cuda 

	}
	return match_counter;
}

__device__ void _migrate(int sorted_position, const unsigned int genes_len,
		int localIndex, unsigned char* g_all_genes, unsigned char* s_genes,
		int* s_fitness, unsigned char* s_expected_genes) {
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
		int max_index = sorted_position - (ISLAND_POPULATION - MIGRATION_SIZE); // from 0 to MIG_SIZE - 1
		unsigned char* genes = &s_genes[localIndex * genes_len];

		//Circular chain of blocks, if block is last, neighbor is the first one.
		memcpy(
				genes, // this genes are one of the worst
				&g_all_genes[(
						blockIdx.x >= NUM_BLOCKS - 1 ? 0 : (blockIdx.x + 1))
						* island_genes_len + max_index * genes_len], // get best genes from neighbor block
				sizeof(unsigned char) * genes_len);

		// update fitness in local
		//s_fitness[localIndex] = fitness_fun(genes, g_expected_genes, genes_len);
	}
}

// framework privates
//Device
__device__ void migrate_best_genes(int localIndex, const unsigned int genes_len,
		int* migrate_flag, int* s_fitness, unsigned char* g_all_genes,
		unsigned char* s_genes, unsigned char* s_expected_genes) {
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
		_migrate(sorted_position, genes_len, localIndex, g_all_genes, s_genes,
				s_fitness, s_expected_genes);
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

		memcpy(son_genes, &s_genes[(mom_or_dad ? mom : dad) * genes_len],
				sizeof(unsigned char) * p1); // first segment, 0 to p1-1, inclusive

		memcpy(&son_genes[p1],
				&s_genes[(mom_or_dad ? dad : mom) * genes_len + p1],
				sizeof(unsigned char) * (p2 - p1)); // second, p1 to p2-1, inc

		memcpy(&son_genes[p2],
				&s_genes[(mom_or_dad ? mom : dad) * genes_len + p2],
				sizeof(unsigned char) * (genes_len - p2)); // third, p2 to genes_len-1

	} else {
		// be equal as dad or mom
		memcpy(son_genes, &s_genes[(mom_or_dad ? mom : dad) * genes_len],
				sizeof(unsigned char) * genes_len);
	}
}

__device__ void mutate(unsigned char* son_genes, unsigned int genes_len,
		curandState_t state) {
	unsigned int random = curand(&state); //32 bits of pure randomness
	unsigned int random2 = curand(&state); //32 bits of pure randomness

	if (MUTATION_CHANCE < random2* UINT_MAX) {
					// each bit has p = 1/genes_len of switch its value
		//make mask with bbits that shoud be switched
		
		float bit_switch_probability = 1.0f / (genes_len*8);

		for (int geneIndex = 0; geneIndex < genes_len; geneIndex++) {
			
			/*if (random <= bit_switch_probability * UINT_MAX) {
				son_genes[geneIndex] = ~son_genes[geneIndex];
			}
			* */
			
			unsigned char switch_mask = 0;
			for (int i = 0; i < 8; i++) {

				unsigned int random = curand(&state); //32 bits of pure randomness
				if (random <= bit_switch_probability * UINT_MAX) {
					switch_mask |= 1;
				}
				switch_mask <<= 1; //1 if switch, 0 if dont
			}
			

			son_genes[geneIndex] ^= switch_mask;
			
			//son_genes[geneIndex] = ~son_genes[geneIndex];
		}
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
	unsigned char* s_expected_genes = (unsigned char*) &migrate_flag[1];

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
		memcpy(s_expected_genes, g_expected_genes,
				sizeof(unsigned char) * genes_len);
		//initial conditions
		*migrate_flag = 0;
}
	__syncthreads();

	/** Evolution loop */
	unsigned int last_gen = 0;

	while (*g_finish_signal == 0 && last_gen++< GENERATION_PER_ROUND) { // todo: while cuando se cambie la flag
		/** Update fitness in Shared memory */

		unsigned char* genes = &s_genes[localIndex * genes_len];
		s_fitness[localIndex] = fitness_fun(genes, s_expected_genes, genes_len);

		/** BEGIN CRITICAL SECTION **/
		if (s_fitness[localIndex] >= genes_len * 8 && *g_finish_signal == 0) { // fitness is number of common bits
			*g_finish_signal = 1;
			memcpy(g_best_genes, genes, sizeof(unsigned char) * genes_len);
			break;
		}
		/** END CRITICAL SECTION **/
		__syncthreads();
		if (*g_finish_signal) {
			printf("HOLY SHITTT\n");
			break;
		}

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
				g_all_genes, s_genes, s_expected_genes);
		__syncthreads();

		// select two parents for new individual, this thread is in charge of that son
		int dad = tournament_selection(TOURNAMENT_K, state, s_fitness);
		int mom = tournament_selection(TOURNAMENT_K, state, s_fitness);


		unsigned char* son_genes = &s_son_genes[localIndex * genes_len];
		crossover(dad, mom, son_genes, s_genes, genes_len, state);
		//son_genes contain crossed over genes of dad and mom
		mutate(son_genes, genes_len, state);
		__syncthreads();

		memcpy(&s_genes[localIndex * genes_len], son_genes,
				sizeof(unsigned char) * genes_len);
	}
	__syncthreads();

	if (localIndex == 0) {
		// one thread copy all the island genes from shared memory to global
		int island_genes_len = genes_len * ISLAND_POPULATION;
		memcpy(&g_all_genes[blockIdx.x * island_genes_len], s_genes,
				sizeof(unsigned char) * island_genes_len);
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
__host__ unsigned char* init_resources() {
	printf("Initializing resources.\n");
	// custom case, image
	//unsigned char* h_expected_genes = file_to_byte_array(
	//		"/home/vichoko/eclipse-workspace/cuda-GA/image.bmp");

	// test with word as genes
	h_expected_genes = (unsigned char*) malloc(
			sizeof(unsigned char) * 32);
	unsigned char perfection[32] = {'s','u','p','e','r','c','a','l','i','f','r','a','g','i','l','y','s','u','p','e','r','c','a','l','i','f','r','a','g','i','l','\0'}; //31 bytes + \0
	memcpy(h_expected_genes, perfection, sizeof(unsigned char) * 32);
	genes_len = 32;
	printf("perfect word is %s\n", h_expected_genes);
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
	return h_expected_genes;
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



//todo: move
unsigned char* d_e_genes;
unsigned char* d_a_genes;
int* ftnss_ptr;
int* d_ftnss_ptr;
__global__ void _fitness_test_kernel(unsigned char* expected_genes,
		unsigned char* actual_genes, const int genes_size, int* ftnss_ptr) {
	if (blockIdx.x == 0) {
		if (threadIdx.x == 0) {
			*ftnss_ptr = fitness_fun(actual_genes, expected_genes, genes_size);
		}
	}
}
void _fitness_test_init_resources(int genes_size) {
	ftnss_ptr = (int*) malloc(sizeof(int));
	gpuErrchk(cudaMalloc(&d_e_genes, genes_size));
	gpuErrchk(cudaMalloc(&d_a_genes, genes_size));
	gpuErrchk(cudaMalloc(&d_ftnss_ptr, sizeof(int)));

}

void _fitness_test_release_resources() {
	//gpuErrchk( cudaFree(&d_e_genes));
	//gpuErrchk( cudaFree(&d_a_genes));
	//gpuErrchk( cudaFree(&d_ftnss_ptr));
}

int _fitness_test_kernel_wrapper(unsigned char* expected_genes,
		unsigned char* actual_genes, const int genes_size) {
	gpuErrchk(
			cudaMemcpy(d_e_genes, expected_genes, genes_size,
					cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpy(d_a_genes, actual_genes, genes_size,
					cudaMemcpyHostToDevice));

	_fitness_test_kernel<<<1, 1>>>(d_e_genes, d_a_genes, genes_size,
			d_ftnss_ptr);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(
			cudaMemcpy(ftnss_ptr, d_ftnss_ptr, sizeof(int),
					cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	return *ftnss_ptr;

}
int _fitness_test() {
	_fitness_test_init_resources(4);
	// fitness use acual_genes, expected_gene, gene_size and calcuates fitness (int; more is better).
	unsigned char expected_genes[4] = { 0xff, 0xff, 0xff, 0xff };
	unsigned char actual_genes_a[4] = { 0, 0, 0, 0 };

	int min_fitness = _fitness_test_kernel_wrapper(actual_genes_a,
			expected_genes, 4);
	if (min_fitness != 0) {
		return -1;
	}

	int max_fitness = _fitness_test_kernel_wrapper(expected_genes,
			expected_genes, 4);
	int max_fitness2 = _fitness_test_kernel_wrapper(actual_genes_a,
			actual_genes_a, 4);

	if (max_fitness != 4 * 8) {
		return -2;
	}
	if (max_fitness != max_fitness2) {
		return -3;
	}
	unsigned char expected_genes_b[4] = { 0xAA, 0xAA, 0xAA, 0xAA }; //10101010
	unsigned char actual_genes_b[4] = { 0x55, 0x55, 0x55, 0x55 }; //01010101

	if (_fitness_test_kernel_wrapper(actual_genes_b, expected_genes_b, 4)
			!= 0) {
		return -4;
	}
	if (_fitness_test_kernel_wrapper(expected_genes_b, expected_genes_b, 4)
			!= 4 * 8) {
		return -5;
	}
	if (_fitness_test_kernel_wrapper(expected_genes_b, expected_genes_b, 4)
			!= _fitness_test_kernel_wrapper(actual_genes_b, actual_genes_b,
					4)) {
		return -6;
	}

	unsigned char expected_genes_c[] = "hola";
	unsigned char actual_genes_c[] = "mola";
	if (_fitness_test_kernel_wrapper(expected_genes_c, actual_genes_c, 5)
			< 4 * 8) {
		return -7;
	}

	if (_fitness_test_kernel_wrapper(expected_genes_c, actual_genes_c, 5)
			== 5 * 8) {
		return -8;
	}

	return 0;
}
//endtodo

__host__ unsigned char* run_evolution() {
	// calculate size of shared memory
	/** need to store island genes and fitnesses **/
	int shared_mem_size = sizeof(unsigned char) * genes_len * ISLAND_POPULATION; // for parent genes
	shared_mem_size += sizeof(unsigned char) * genes_len * ISLAND_POPULATION; // for son genes
	shared_mem_size += sizeof(unsigned char) * genes_len; // for expected genes
	shared_mem_size += sizeof(int) * ISLAND_POPULATION; // for fitnesses
	shared_mem_size += sizeof(int); // for migration flag

	unsigned char* all_genes = (unsigned char*) malloc(sizeof(unsigned char)* genes_len* TOTAL_POPULATION);
	unsigned char* best_genes;
	_fitness_test_init_resources(genes_len);

	unsigned int verbose = 1;
	printf("Starting evolution\n");
	int finish = 0;
	int round = 0;
	while(!finish){
		round++;
		island_controller<<<
				NUM_BLOCKS, 	// NUMBER OF ISLANDS
				THREADS_PER_BLOCK, // ISLAND_POPULATION
				shared_mem_size // tamaño de memoria compartida
		>>>(genes_len, d_all_genes, d_best_genes, d_finish_signal, time(NULL), // random seed
		verbose, //todo: delete
				d_expected_genes); //custom param
		gpuErrchk(cudaThreadSynchronize());
		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk( cudaMemcpy(all_genes, d_all_genes, sizeof(unsigned char)* genes_len* TOTAL_POPULATION, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaThreadSynchronize());

		int max_fit = 0;
		for (int indiv_index= 0; indiv_index< TOTAL_POPULATION; indiv_index++){
			unsigned char* actual_genes = &all_genes[indiv_index* genes_len];
			int fit = _fitness_test_kernel_wrapper(actual_genes,h_expected_genes, genes_len);
			if (fit> max_fit){
				max_fit = fit;
				best_genes = actual_genes;
			}
		}
		best_genes[genes_len-1] = '\0';
		printf("gen: %d, best_fit: %d, perf_fit: %d; the word is %s\n", round*GENERATION_PER_ROUND, max_fit, genes_len*8, best_genes);
		if (max_fit == genes_len*8){
			finish = 1;
		}

	}
	//gpuErrchk(cudaDeviceSynchronize());
	// if im here, perfect individual should exist


	gpuErrchk(
			cudaMemcpy(best_genes, d_best_genes,
					sizeof(unsigned char) * genes_len, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaThreadSynchronize());

	printf("	Evolution finished.\n");
	return best_genes; // Todo: asegurar que sea el individuo perfecto
}

/** TESTING **/

/** PROCESS OVERALL:
 * 1. GENERATE POPULATION
 * 2. BEGIN EVOLUTION LOOP
 * 	2.1. MIGRATE w/ MIGRATION CHANCE
 * 	2.2. PICK 2 PARENTS w/ TOURNAMENT SELECTION (k= 3)
 * 	2.3. CROSSOVER w/ CROSSOVER CHANCE => 1 CHILD
 * 	2.4. MUTATE w/ MUTATION CHANCE
 * 	2.5. CHECK IF CHILD IS PERFECT, END IF DO
 */

unsigned int genes_len_test;

size_t genes_size;
size_t total_genes_size;
size_t shared_mem_size;

unsigned char* all_genes_test;
unsigned char* d_all_genes_test;

int __init_test_resources() {
	genes_len_test = 100;

	shared_mem_size =
			sizeof(unsigned char) * genes_len_test * ISLAND_POPULATION; // for parent genes
	shared_mem_size += sizeof(unsigned char) * genes_len_test
			* ISLAND_POPULATION; // for son genes
	shared_mem_size += sizeof(int) * ISLAND_POPULATION; // for fitnesses
	shared_mem_size += sizeof(int); // for migration flag

	genes_size = sizeof(unsigned char) * genes_len_test;
	total_genes_size = genes_size * TOTAL_POPULATION;

	all_genes_test = (unsigned char*) malloc(total_genes_size);
	gpuErrchk(cudaMalloc(&d_all_genes_test, total_genes_size));

	return 0;
}

int population_test() {
	generate_initial_population<<<
	NUM_BLOCKS,
	ISLAND_POPULATION>>>(genes_len_test, d_all_genes_test, time(NULL));
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(
			cudaMemcpy(all_genes_test, d_all_genes_test, total_genes_size,
					cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// every byte should be a valid one
	for (int i = 0; i < total_genes_size / (sizeof(unsigned char)); i++) {
		//printf("%uc\n", all_genes_test[i]);
		if (all_genes_test[i] > 255) {
			return -1;
		}
	}
	return 0;
}

__global__ void _migration_test_kernel(const unsigned int genes_len,
		unsigned char* g_all_genes, unsigned char* g_this_genes,
		const unsigned int island_id) {
	//initialization
	extern __shared__ int shared_array[];
	int* s_fitness = shared_array;
	unsigned char* s_genes = (unsigned char*) &s_fitness[ISLAND_POPULATION];
	unsigned char* s_son_genes = (unsigned char*) &s_genes[ISLAND_POPULATION
			* genes_len];
	int* migrate_flag = (int*) &s_son_genes[ISLAND_POPULATION * genes_len];
	int island_genes_len = genes_len * ISLAND_POPULATION;

	if (blockIdx.x == island_id) {

		//int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; // indice del individuo/thread
		int localIndex = threadIdx.x;
		// end initilaization

		/* copy genes from Global Memory (generate_initial_population) to Shared */
		if (localIndex == 0) {
			// one thread copy all the island genes to shared memory
			memcpy(s_genes, &g_all_genes[blockIdx.x * island_genes_len],
					sizeof(unsigned char) * island_genes_len);

			//g_this_genes[0] = 255;

			//initial conditions
		}
		__syncthreads();

		// modify the MIGRATION_SIZE first individuals
		if (localIndex < MIGRATION_SIZE) {
			for (int i = 0; i < genes_len; i++) {
				s_genes[localIndex * genes_len + i] = localIndex % 256;
			}
		}
		*migrate_flag = 1;
		__syncthreads();

		_migrate(localIndex, genes_len, localIndex, g_all_genes, s_genes,
				s_fitness,
				NULL);

		__syncthreads();
		if (localIndex == 0) {
			memcpy(g_this_genes, s_genes,
					sizeof(unsigned char) * island_genes_len);
		}
		__syncthreads();
	}

}
int _migration_test() {

	// shared vs global device memory consistency test
	unsigned int island_id_a = 1;
	unsigned char* h_a_genes = (unsigned char*) malloc(
			genes_size * ISLAND_POPULATION);
	unsigned char* d_a_genes;
	gpuErrchk(cudaMalloc(&d_a_genes, genes_size* ISLAND_POPULATION));

	_migration_test_kernel<<<NUM_BLOCKS, 	// NUMBER OF ISLANDS
			ISLAND_POPULATION, // ISLAND_POPULATION
			shared_mem_size // tamaño de memoria compartida
	>>>(genes_len_test, d_all_genes_test, d_a_genes, island_id_a);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(
			cudaMemcpy( h_a_genes, d_a_genes, genes_size* ISLAND_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(
			cudaMemcpy(all_genes_test, d_all_genes_test, genes_size* TOTAL_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// check if last MIGRATION_SIZE* genes_len_test elements are equal al neighbor first elements
	for (int i = 0; i < MIGRATION_SIZE * genes_len_test; i++) {
		if (h_a_genes[(ISLAND_POPULATION - MIGRATION_SIZE) * genes_len_test + i]
				!= all_genes_test[(island_id_a + 1) * genes_len_test
						* ISLAND_POPULATION + i]) {
			return -1;
		}
	}

	// shared vs global test
	for (int i = 0; i < MIGRATION_SIZE * genes_len_test; i++) {
		if (h_a_genes[i]
				!= all_genes_test[island_id_a * genes_len_test
						* ISLAND_POPULATION + i]) {
			return 1;
		}
	}

	// edit persistence in global memory test
	unsigned int island_id_b = 0;
	unsigned char* h_b_genes = (unsigned char*) malloc(
			genes_size * ISLAND_POPULATION);
	unsigned char* d_b_genes;
	gpuErrchk(cudaMalloc(&d_b_genes, genes_size* ISLAND_POPULATION));
	_migration_test_kernel<<<NUM_BLOCKS, 	// NUMBER OF ISLANDS
			ISLAND_POPULATION, // ISLAND_POPULATION
			shared_mem_size // tamaño de memoria compartida
	>>>(genes_len_test, d_all_genes_test, d_b_genes, island_id_b);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(
			cudaMemcpy( h_b_genes, d_b_genes, genes_size* ISLAND_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(
			cudaMemcpy(all_genes_test, d_all_genes_test, genes_size* TOTAL_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// check if last MIGRATION_SIZE* genes_len_test elements are equal al neighbor first elements just modified
	for (int i = 0; i < MIGRATION_SIZE * genes_len_test; i++) {
		if (h_b_genes[(ISLAND_POPULATION - MIGRATION_SIZE) * genes_len_test + i]
				!= h_a_genes[i]) {
			return -2;
		}
	}

	//border case test
	unsigned int island_id_c = 0;
	unsigned char* h_c_genes = (unsigned char*) malloc(
			genes_size * ISLAND_POPULATION);
	unsigned char* d_c_genes;
	gpuErrchk(cudaMalloc(&d_c_genes, genes_size* ISLAND_POPULATION));
	_migration_test_kernel<<<NUM_BLOCKS, 	// NUMBER OF ISLANDS
			ISLAND_POPULATION, // ISLAND_POPULATION
			shared_mem_size // tamaño de memoria compartida
	>>>(genes_len_test, d_all_genes_test, d_c_genes, island_id_c);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(
			cudaMemcpy( h_c_genes, d_c_genes, genes_size* ISLAND_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(
			cudaMemcpy(all_genes_test, d_all_genes_test, genes_size* TOTAL_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// check if last MIGRATION_SIZE* genes_len_test elements are equal al neighbor first elements just modified
	for (int i = 0; i < MIGRATION_SIZE * genes_len_test; i++) {
		if (h_c_genes[(ISLAND_POPULATION - MIGRATION_SIZE) * genes_len_test + i]
				!= h_b_genes[i]) {
			return -3;
		}

		if (h_c_genes[(ISLAND_POPULATION - MIGRATION_SIZE) * genes_len_test + i]
				!= all_genes_test[i]) {
			return -4;
		}
	}

	return 0;
}
int tournament_selection_test() {

	return 0;

}

__global__ void _crossover_test_kernel(const unsigned int genes_len,
		unsigned char* g_all_genes, unsigned int seed,
		unsigned char* g_this_genes, const unsigned int island_id) {
	if (blockIdx.x == island_id) {
		extern __shared__ int shared_array[];
		int* s_fitness = shared_array;
		unsigned char* s_genes = (unsigned char*) &s_fitness[ISLAND_POPULATION];
		unsigned char* s_son_genes = (unsigned char*) &s_genes[ISLAND_POPULATION
				* genes_len];
		int* migrate_flag = (int*) &s_son_genes[ISLAND_POPULATION * genes_len];

		int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; // indice del individuo/thread
		int localIndex = threadIdx.x;
		int island_genes_len = genes_len * ISLAND_POPULATION;

		/* CUDA's random number library */
		curandState_t state;
		curand_init(seed, /* the seed controls the sequence of random values that are produced */
		(globalIndex + 1), /* the sequence number is only important with multiple cores */
		genes_len, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

		/* copy genes from Global Memory (generate_initial_population) to Shared */
		if (localIndex == 0) {
			// one thread copy all the island genes to shared memory
			memcpy(s_genes, &g_all_genes[blockIdx.x * island_genes_len],
					sizeof(unsigned char) * island_genes_len);
			//initial conditions
			*migrate_flag = 0;
		}

		//unsigned char* genes = &s_genes[localIndex * genes_len];
		//s_fitness[localIndex] = fitness_fun(genes, g_expected_genes, genes_len);
		__syncthreads();

		// select two parents for new individual, this thread is in charge of that son
		//int dad = tournament_selection(3, state, s_fitness);
		//int mom = tournament_selection(3, state, s_fitness);
		int dad = localIndex; //has to be deterministic for testing
		int mom = (localIndex < blockDim.x - 1) ? localIndex + 1 : 0;
		__syncthreads();

		unsigned char* son_genes = &s_son_genes[localIndex * genes_len];
		crossover(dad, mom, son_genes, s_genes, genes_len, state);
		//son_genes contain crossed over genes of dad and mom
		__syncthreads();
		memcpy(&s_genes[localIndex * genes_len], son_genes,
				sizeof(unsigned char) * genes_len);
		__syncthreads();
		if (localIndex == 0) {
			memcpy(g_this_genes, s_genes,
					sizeof(unsigned char) * island_genes_len);
		}
		__syncthreads();

	}
}

int _crossover_test() {
	// shared vs global device memory consistency test
	unsigned int island_id_a = 0;
	unsigned char* h_a_genes = (unsigned char*) malloc(
			genes_size * ISLAND_POPULATION);
	unsigned char* d_a_genes;
	gpuErrchk(cudaMalloc(&d_a_genes, genes_size* ISLAND_POPULATION));

	_crossover_test_kernel<<<NUM_BLOCKS, 	// NUMBER OF ISLANDS
			ISLAND_POPULATION, // ISLAND_POPULATION
			shared_mem_size // tamaño de memoria compartida
	>>>(genes_len_test, d_all_genes_test, time(NULL), d_a_genes, island_id_a);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(
			cudaMemcpy( h_a_genes, d_a_genes, genes_size* ISLAND_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(
			cudaMemcpy(all_genes_test, d_all_genes_test, genes_size* TOTAL_POPULATION, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	for (int individual = 0; individual < ISLAND_POPULATION; individual++) {
		for (int gene = 0; gene < genes_len_test; gene++) {
			int son_index = individual * genes_len_test;
			int dad_index = (ISLAND_POPULATION * island_id_a + individual)
					* genes_len_test;
			int mom_index =
					individual >= ISLAND_POPULATION - 1 ?
							0 :
							(ISLAND_POPULATION * island_id_a + individual + 1)
									* genes_len_test;

			if (h_a_genes[son_index + gene] != all_genes_test[dad_index + gene]
					&& h_a_genes[individual * genes_len_test + gene]
							!= all_genes_test[mom_index + gene]) {
				return -1;
			}
		}

	}

	return 0;
}
int mutation_test() {

	return 0;
}





int __run_tests() {
	int err = 0;

	if ((err = __init_test_resources())) {
		printf("__init_test_resources error %d\n", err);
		exit(err);
	}
	printf("pass init_resources\n");
	if ((err = population_test())) {
		printf("population_test error %d\n", err);
		exit(err);
	}
	printf("pass population_test\n");

	if ((err = _migration_test())) {
		printf("migration_test error %d\n", err);
		exit(err);
	}
	printf("pass migration_test\n");

	if ((err = tournament_selection_test())) {
		printf("tournament_selection_test error %d\n", err);
		exit(err);
	}
	printf("pass tournament_selection_test\n");

	if ((err = _crossover_test())) {
		printf("crossover_test error %d\n", err);
		exit(err);
	}
	printf("pass crossover_test\n");

	if ((err = mutation_test())) {
		printf("mutation_test error %d\n", err);
		exit(err);
	}
	printf("pass mutation_test\n");

	if ((err = _fitness_test())) { //todo: check if fitness gets better, if t higher comapring withe xpected shit you kno
		printf("fitness_test error %d\n", err);
		exit(err);
	}
	printf("pass mutation_test\n");

	if (err == 0) {
		printf("all pass\n");
	}
	return err;
}

// MAIN
// CPU Controller
int main() {
	cudaSetDevice(1);
	int test_mode = 0;
	if (!test_mode) {
		printf("Started CUDA GA.\n");
		unsigned char* expected_genes = init_resources();
		init_population();

		unsigned char* evoluted_genes = run_evolution();

		_fitness_test_init_resources(genes_len);
		int fitness = _fitness_test_kernel_wrapper(evoluted_genes,
				expected_genes, genes_len);

		printf("fit: %d, perfect word is %s, evoluted one is %s\n", fitness, expected_genes,
				evoluted_genes);
		//FILE* file = fopen("evoluted_img.bmp", "wb");
		//fwrite(evoluted_genes, 1, genes_len, file);

	} else {
		printf("Started CUDA GA - TEST SUITE\n");
		int res = __run_tests();
		printf("Finished tests\n");
		return res;
	}

}
