

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define DIES 0
#define ALIVE 1
//#define DEBUG2 1

/* function to measure time taken */
double gettime(void) 
{
    struct timeval tval;

    gettimeofday(&tval, NULL);

    return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

void printarray(int *a, int M, int N, FILE *fp) 
{
    int i, j;
    for (i = 0; i < M+2; i++) 
    {
        for (j = 0; j< N+2; j++)
            fprintf(fp, "%d ", a[i*(N+2) + j]);
        fprintf(fp, "\n");
    }
}

int check_array(int *a, int M, int N) 
{
    int value=0;
    for (int i = 1; i < M+1; i++)
        for (int j = 1; j < N+1; j++)
            value += a[i*(N+2) + j];
    return value;
}

/* cuda kerel to compute a step in the game */
__global__ void compute_kernel(int *life, int *temp, int N) 
{

    extern __shared__ int life_shared[];

    int i = blockIdx.x + 1;
    int j = threadIdx.x + 1;

    /* copy three rows of life matrix into shared memory */
    for (int k = j; k < N+1; k += blockDim.x) 
    {
        life_shared[k] = life[(i-1)*(N+2) + k];
        life_shared[N+2 + k] = life[(i)*(N+2) + k];
        life_shared[2*(N+2) + k] = life[(i+1)*(N+2) + k];


    }

    /* copy border values to shared memory*/
    if (threadIdx.x == 0) 
    {
        life_shared[0] = life[(i-1)*(N+2)];
        life_shared[N+2] = life[(i)*(N+2)];
        life_shared[2*(N+2)] = life[(i+1)*(N+2)];

        life_shared[N+1] = life[(i-1)*(N+2) + N+1];
        life_shared[N+2 + N+1] = life[(i)*(N+2) + N+1];
        life_shared[2*(N+2) + N+1] = life[(i+1)*(N+2) + N+1];
    }

    __syncthreads();

    for (int k = j; k < N+1; k += blockDim.x) 
    {
        /* find out the value of the current cell */
        int value = life_shared[(k-1)] + life_shared[k] + life_shared[(k+1)] + 
                    life_shared[(N+2) + (k-1)] + life_shared[(N+2) + (k+1)] + 
                    life_shared[2*(N+2) + (k-1)] + life_shared[2*(N+2) + k] + life_shared[2*(N+2) + (k+1)] ;
        
        /* check if the cell dies or life is born */
        if (life_shared[(N+2) + k]) 
        { // cell was alive in the earlier iteration
            if (value < 2 || value > 3) 
            {
                temp[i*(N+2) + k] = DIES ;
            }
            else // value must be 2 or 3, so no need to check explicitly
                temp[i*(N+2) + k] = ALIVE ; // no change
        } 
        else 
        { // cell was dead in the earlier iteration
            if (value == 3) 
            {
                temp[i*(N+2) + k] = ALIVE;
            }
            else
                temp[i*(N+2) + k] = DIES; // no change
        }
    }
}

void compute(int *life, int *temp, int M, int N) 
{

    /* set number of threads and number of blocks for cuda kernel */
    int numThreads = 1024;
    int numBlocks =N;
 
    /* set the dyamic amount of shared memory to be reserved per block*/
    unsigned sharedMemSize = (3 * (N+2) * sizeof(int));

    compute_kernel <<<numBlocks, numThreads, sharedMemSize>>> (life, temp, N);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
 
int main(int argc, char **argv) 
{
    int N, NTIMES, *life=NULL, *temp=NULL;
    int i, j, k;
    double t1, t2;

#if defined(DEBUG1) || defined(DEBUG2)
    FILE *fp;
    char GOL[32];
#endif
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    N = 10000;
    NTIMES = 5000;

    /* Allocate memory for both arrays */
    life = (int*) malloc((N+2)*(N+2)*sizeof(int));
    temp = (int*) malloc((N+2)*(N+2)*sizeof(int));

    /* Initialize the boundaries of the life matrix */
    for (i = 0; i < N+2; i++) 
    {
        life[i*(N+2)] = life[i*(N+2) + (N+1)] = DIES ;
        temp[i*(N+2)] = temp[i*(N+2) + (N+1)] = DIES ;
    }
    for (j = 0; j < N+2; j++) 
    {
        life[j] = life[(N+1)*(N+2) + j] = DIES ;
        temp[j] = temp[(N+1)*(N+2) + j] = DIES ;
    }

    /* Initialize the life matrix */
    for (i = 1; i < N+1; i++) 
    {
        for (j = 1; j< N+1; j++) 
        {
            if (drand48() < 0.5) 
	            life[i*(N+2) + j] = ALIVE ;
            else
	            life[i*(N+2) + j] = DIES ;
        }
    }

#ifdef DEBUG1
    /* Display the initialized life matrix */
    printf("Printing to file: output.%d.0\n",N);
    sprintf(GOL,"output.%d.0",N);
    fp = fopen("GOL.txt", "w");
    printarray(life, N, N, fp);
    fclose(fp);
#endif

    /* Define device memory pointers */
    int *life_device, *temp_device;

    /* Allocate global memory in device */
    cudaMalloc(&life_device, (N+2)*(N+2)*sizeof(int));
    cudaMalloc(&temp_device, (N+2)*(N+2)*sizeof(int));

    /* Initialize the device pointers */
    cudaMemcpy(life_device, life, (N+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_device, temp, (N+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);

    t1 = gettime();
    cudaEventRecord(start);

    /* Play the game of life for given number of iterations */
    for (k = 0; k < NTIMES; k += 2) 
    {
        compute(life_device, temp_device, N, N);
        compute(temp_device, life_device, N, N);
    }
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Taken by Kernal: %6f milisec \n",milliseconds);
    t2 = gettime();

    /* Copy the result back to host */
    cudaMemcpy(life, life_device, (N+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp, temp_device, (N+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost);

    int life_remaining = check_array(life, N, N);
    printf("Total Time taken for size = %d after %d iterations = %f sec\n", N, k, t2-t1);
    printf("No. of cells alive after %d iterations = %d\n", k, life_remaining);

#ifdef DEBUG2
    /* Display the life matrix after k iterations */
    printf("Printing to file: output.%d.%d\n",N,k);
    sprintf(GOL,"output.%d.%d",N,k);
    fp = fopen("GOL.txt", "w");
    printarray(life, N, N, fp);
    fclose(fp);
#endif

    cudaFree(temp_device);
    cudaFree(life_device);
    free(life);
    free(temp);

    return 0;
}
