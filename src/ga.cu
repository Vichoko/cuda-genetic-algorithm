#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#include "ga.h"

// MACROS
#define NUM_BLOCKS 500
// GEFORCE GTX 960m has 640 blocks.
#define THREADS_PER_BLOCK 128
// Island is a block. Every individual is a thread.
#define TOTAL_POPULATION NUM_BLOCKS*THREADS_PER_BLOCK
// azucar sintactico
#define ISLAND_POPULATION THREADS_PER_BLOCK
// azucar sintactico

// GLOBALS
unsigned char* d_all_genes;
unsigned int* d_finish_signal;
unsigned char* d_best_genes;

// Esquema implica modificar funcion de fitness y tamaño de gen, para adaptarlo al problema.
unsigned int genes_len = 100; // TODO: size should be image size in bytes. ¿que pasa si imagen no termian en byte completo?

__device__ int fitness(unsigned char* actual_genes, int genes_len){
	unsigned char* expected_genes; // TODO: this should be the target BMP image.
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
__global__ void island_controller(const unsigned int genes_len, unsigned char* all_genes, unsigned int seed){
	//
}

__global__ void generate_initial_population(const unsigned int genes_len, unsigned char* all_genes, unsigned int seed){
	// cada thread se encarga de generarse a si mismo y colocarse en memoria principal de GPU.
	int globalIndex = blockIdx.x* blockDim.x+ threadIdx.x; // indice del individuo/thread
	int localIndex = threadIdx.x;

	extern __shared__ unsigned char new_genes[];
	// TODO: generate array of genes_len random bytes

	/* CUDA's random number library uses curandState_t to keep track of the seed value
	     we will store a random state for every thread  */
	  curandState_t state;

	for (int geneIndex= 0; geneIndex< genes_len; geneIndex++){
		  /* we have to initialize the state */
		  curand_init(seed, /* the seed controls the sequence of random values that are produced */
				  	  (globalIndex+ 1)*  geneIndex, /* the sequence number is only important with multiple cores */
		              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		              &state);
		  // se debe poder optimizar
		  /* curand works like rand - except that it takes a state as a parameter */
		  unsigned int result = curand(&state);

		  new_genes[geneIndex+ localIndex]= (unsigned char) result% 256; // write in shared memory
		  all_genes[geneIndex+ globalIndex] = (unsigned char) result% 256; // write in global device memory
	}
}

__host__ void init_resources(){
	cudaMalloc(&d_all_genes, sizeof(unsigned char)*genes_len*TOTAL_POPULATION); // arreglo de genes aplanado inicialmente vacio
	cudaMalloc(&d_best_genes, sizeof(unsigned char)*genes_len);
	cudaMalloc(&d_finish_signal, sizeof(int));
	int aux = 0;
	cudaMemcpy(&d_finish_signal, &aux, sizeof(int), cudaMemcpyHostToDevice);
}

__host__ void init_population() {
	printf("Generating initial population.\n");
	generate_initial_population<<<NUM_BLOCKS,
	THREADS_PER_BLOCK, sizeof(unsigned char) * genes_len * ISLAND_POPULATION // tamaño de memoria compartida
	>>>(genes_len, d_all_genes, time(NULL));
	cudaDeviceSynchronize();
	printf("	done Generating initial population.\n");
}

__host__ unsigned char* run_evolution() {
	printf("Starting evolution!\n");
	island_controller<<<NUM_BLOCKS,
	THREADS_PER_BLOCK, sizeof(unsigned char) * genes_len * ISLAND_POPULATION // tamaño de memoria compartida
	>>>(genes_len, d_all_genes, time(NULL));
	cudaDeviceSynchronize();
	printf("Evolution finished.\n");
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
