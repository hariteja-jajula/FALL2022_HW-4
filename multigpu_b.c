#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE 840
#define NDIMS 2
#define GENERATIONS 1
#define BLOCKS

void initialState(int rows, int columns, char *first_generation, int seed, MPI_Comm *cartesian2D, int rank);
void printGrid(int rows, int columns, char *life);
void swap(char** a, char** b);
void nextGeneration(char* d_life, char* d_life_copy, int rows, int columns);
void nextGenerationInner(char* d_life, char* d_life_copy, int rows, int columns);
void nextGenerationOuter(char* d_life, char* d_life_copy, int rows, int columns);
cudaError_t initializeOnDevice(char** d_life, char** d_life_copy, char** h_life, int rows, int columns);
void getGPUCount(int* numOfGPUs);
int
main()
{


   int dim_size[NDIMS], periods[NDIMS], coords[NDIMS];
   int north[NDIMS], east[NDIMS], south[NDIMS], west[NDIMS],
       northeast[NDIMS], southeast[NDIMS], southwest[NDIMS], northwest[NDIMS];


   int reorder, rank, processes, rows, columns, seed;
   int north_rank, east_rank, south_rank, west_rank,
       northeast_rank, southeast_rank, southwest_rank, northwest_rank;
   MPI_Comm cartesian2D;


   MPI_Datatype row_datatype, column_datatype;
   MPI_Request  receive_requests_even[8], send_requests_even[8], receive_requests_odd[8], send_requests_odd[8];
   MPI_Request* send; 
   MPI_Request* recv; 
   MPI_Status   statuses[8];
   double       t1, t2, root;

   // Our Cartesian topology will be a torus, so both fields of "periods" array will have a value of 1
   periods[0] = periods[1] = 1;

   // We will allow MPI to efficiently reorder the processes among the different processors
   reorder = 1;

   // Retrieve the number of available GPUs on the node
   int numGPUs = 0;
   getGPUCount(&numGPUs);

   // Initialize MPI
   MPI_Init(NULL, NULL);
   MPI_Pcontrol(0);
   MPI_Comm_size(MPI_COMM_WORLD, &processes);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef BLOCKS
   // If the number of processes is a perfect square, arrange them evenly in a NXN fashion. Otherwise, there are no restrictions
   root = sqrt((double)processes);

   if (root == floor(root))
       dim_size[0] = dim_size[1] = (int)root;
   else
       dim_size[0] = dim_size[1] = 0;
#else
   dim_size[0] = processes;
   dim_size[1] = 1;
#endif
   if (MPI_Dims_create(processes, NDIMS, dim_size) != MPI_SUCCESS) {
      if (rank == 0) {
         printf("Number of processes and size of grid do not match. MPI_Dims_create() returned an error. Exiting.\n");
      }
      MPI_Abort(MPI_COMM_WORLD, -1);
      MPI_Finalize();
      return -1;
   }

   // Create a 2D Cartesian topology. Find the rank and coordinates of each process
   MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dim_size, periods, reorder, &cartesian2D);
   MPI_Cart_coords(cartesian2D, rank, NDIMS, coords);

   // We add 2 to each dimension in order to include the halo rows and columns
   rows = (SIZE / dim_size[0]) + 2;
   columns = (SIZE /dim_size[1]) + 2;

   // Calculate the coordinates and ranks of all neighbors
   north[0] = coords[0] - 1;
   north[1] = coords[1];
   MPI_Cart_rank(cartesian2D, north, &north_rank);

   east[0] = coords[0];
   east[1] = coords[1] + 1;
   MPI_Cart_rank(cartesian2D, east, &east_rank);

   south[0] = coords[0] + 1;
   south[1] = coords[1];
   MPI_Cart_rank(cartesian2D, south, &south_rank);

   west[0] = coords[0];
   west[1] = coords[1] - 1;
   MPI_Cart_rank(cartesian2D, west, &west_rank);

   northeast[0] = coords[0] - 1;
   northeast[1] = coords[1] + 1;
   MPI_Cart_rank(cartesian2D, northeast, &northeast_rank);

   southeast[0] = coords[0] + 1;
   southeast[1] = coords[1] + 1;
   MPI_Cart_rank(cartesian2D, southeast, &southeast_rank);

   southwest[0] = coords[0] + 1;
   southwest[1] = coords[1] - 1;
   MPI_Cart_rank(cartesian2D, southwest, &southwest_rank);

   northwest[0] = coords[0] - 1;
   northwest[1] = coords[1] - 1;
   MPI_Cart_rank(cartesian2D, northwest, &northwest_rank);

   // We need two datatypes for the halos, one for the rows and one for the columns
   MPI_Type_contiguous(columns - 2, MPI_CHAR, &row_datatype);
   MPI_Type_commit(&row_datatype);

   MPI_Type_vector(rows - 2, 1, columns, MPI_CHAR, &column_datatype);
   MPI_Type_commit(&column_datatype);

   // Initialize the arrays that will be used inside the main loop
   recv = (MPI_Request*)malloc(8 * sizeof(MPI_Request));
   send = (MPI_Request*)malloc(8 * sizeof(MPI_Request));

   // Pointers to our 2D grid, and its necessary copy
   char* h_life = (char*)malloc(rows * columns * sizeof(char));
   char* d_life;
   char* d_life_copy;

   // Generate the first generation according to the random seed
   seed = rank + 2;
   initialState(rows, columns, h_life, seed, &cartesian2D, rank);

   // Assign a GPU to each MPI process and then initialize the
   // Game of Life boards on the GPU too
   cudaSetDevice(rank % numGPUs);

   if (initializeOnDevice(&d_life, &d_life_copy, &h_life, rows, columns) != cudaSuccess) {
      fprintf(stderr, "Could not initialize on the device. Exiting.\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
      MPI_Finalize();
      return -1;
   }


   /* We implement persistent communication, since the neighboring processes will always remain the same through the execution of the program */
   /* These are for the even iterations of the loop, e.g. generation = 0, 2, 4, 6, 8 etc.                                                     */

   MPI_Recv_init(d_life + 1, 1, row_datatype, north_rank, north_rank, cartesian2D, &receive_requests_even[0]);
   MPI_Recv_init(d_life + (rows - 1) * columns + 1, 1, row_datatype, south_rank, south_rank, cartesian2D, &receive_requests_even[1]);
   MPI_Recv_init(d_life + columns, 1, column_datatype, west_rank, west_rank, cartesian2D, &receive_requests_even[2]);
   MPI_Recv_init(d_life + (columns * 2) - 1, 1, column_datatype, east_rank, east_rank, cartesian2D, &receive_requests_even[3]);

   MPI_Recv_init(d_life, 1, MPI_CHAR, northwest_rank, northwest_rank, cartesian2D, &receive_requests_even[4]);
   MPI_Recv_init(d_life + columns - 1, 1, MPI_CHAR, northeast_rank, northeast_rank, cartesian2D, &receive_requests_even[5]);
   MPI_Recv_init(d_life + columns * (rows - 1), 1, MPI_CHAR, southwest_rank, southwest_rank, cartesian2D, &receive_requests_even[6]);
   MPI_Recv_init(d_life + (columns * rows) - 1, 1, MPI_CHAR, southeast_rank, southeast_rank, cartesian2D, &receive_requests_even[7]);

   MPI_Send_init(d_life + (rows - 2) * columns + 1, 1, row_datatype, south_rank, rank, cartesian2D, &send_requests_even[0]);
   MPI_Send_init(d_life + columns + 1, 1, row_datatype, north_rank, rank, cartesian2D, &send_requests_even[1]);
   MPI_Send_init(d_life + (columns * 2) - 2, 1, column_datatype, east_rank, rank, cartesian2D, &send_requests_even[2]);
   MPI_Send_init(d_life + columns + 1, 1, column_datatype, west_rank, rank, cartesian2D, &send_requests_even[3]);

   MPI_Send_init(d_life + columns * (rows - 1) - 2, 1, MPI_CHAR, southeast_rank, rank, cartesian2D, &send_requests_even[4]);
   MPI_Send_init(d_life + columns * (rows - 2) + 1, 1, MPI_CHAR, southwest_rank, rank, cartesian2D, &send_requests_even[5]);
   MPI_Send_init(d_life + (columns * 2) - 2, 1, MPI_CHAR, northeast_rank, rank, cartesian2D, &send_requests_even[6]);
   MPI_Send_init(d_life + columns + 1, 1, MPI_CHAR, northwest_rank, rank, cartesian2D, &send_requests_even[7]);

   /**************************************************************************************/
   /* These are for the odd iterations of the loop, e.g. generation = 1, 3, 5, 7, 9 etc. */
   /**************************************************************************************/
   MPI_Recv_init(d_life_copy + 1, 1, row_datatype, north_rank, north_rank, cartesian2D, &receive_requests_odd[0]);
   MPI_Recv_init(d_life_copy + (rows - 1) * columns + 1, 1, row_datatype, south_rank, south_rank, cartesian2D, &receive_requests_odd[1]);
   MPI_Recv_init(d_life_copy + columns, 1, column_datatype, west_rank, west_rank, cartesian2D, &receive_requests_odd[2]);
   MPI_Recv_init(d_life_copy + (columns * 2) - 1, 1, column_datatype, east_rank, east_rank, cartesian2D, &receive_requests_odd[3]);

   MPI_Recv_init(d_life_copy, 1, MPI_CHAR, northwest_rank, northwest_rank, cartesian2D, &receive_requests_odd[4]);
   MPI_Recv_init(d_life_copy + columns - 1, 1, MPI_CHAR, northeast_rank, northeast_rank, cartesian2D, &receive_requests_odd[5]);
   MPI_Recv_init(d_life_copy + columns * (rows - 1), 1, MPI_CHAR, southwest_rank, southwest_rank, cartesian2D, &receive_requests_odd[6]);
   MPI_Recv_init(d_life_copy + (columns * rows) - 1, 1, MPI_CHAR, southeast_rank, southeast_rank, cartesian2D, &receive_requests_odd[7]);

   MPI_Send_init(d_life_copy + (rows - 2) * columns + 1, 1, row_datatype, south_rank, rank, cartesian2D, &send_requests_odd[0]);
   MPI_Send_init(d_life_copy + columns + 1, 1, row_datatype, north_rank, rank, cartesian2D, &send_requests_odd[1]);
   MPI_Send_init(d_life_copy + (columns * 2) - 2, 1, column_datatype, east_rank, rank, cartesian2D, &send_requests_odd[2]);
   MPI_Send_init(d_life_copy + columns + 1, 1, column_datatype, west_rank, rank, cartesian2D, &send_requests_odd[3]);

   MPI_Send_init(d_life_copy + columns * (rows - 1) - 2, 1, MPI_CHAR, southeast_rank, rank, cartesian2D, &send_requests_odd[4]);
   MPI_Send_init(d_life_copy + columns * (rows - 2) + 1, 1, MPI_CHAR, southwest_rank, rank, cartesian2D, &send_requests_odd[5]);
   MPI_Send_init(d_life_copy + (columns * 2) - 2, 1, MPI_CHAR, northeast_rank, rank, cartesian2D, &send_requests_odd[6]);
   MPI_Send_init(d_life_copy + columns + 1, 1, MPI_CHAR, northwest_rank, rank, cartesian2D, &send_requests_odd[7]);

   // Synchronize all the processes before we start
   MPI_Barrier(cartesian2D);
   t1 = MPI_Wtime();
   MPI_Pcontrol(1);

#ifdef DEBUG_COORDINATES
   // Print the coordinates and neighbours of process 0 to get a general idea about the layout of the grid
   if (rank == 0) {
      printf("rows are %d\n", rows);
      printf("columns are %d\n\n", columns);
      printf("The cartesian topology for process 0 is\n");
      printf("%d, %d\n", coords[0], coords[1]);
      printf("north %d\n", north_rank);
      printf("east %d\n", east_rank);
      printf("west %d\n", west_rank);
      printf("south %d\n", south_rank);
   }
#endif

#ifdef DEBUG_GRID
   MPI_Status statusTemp;
#ifdef BLOCKS
   // Print the grid of every process, before the exchange of the halo elements and before the beginning of the main loop
   if (rank == 0) {
      char* temp = (char*)malloc(rows * columns * sizeof(char));

      printf("The grid for process 0 is:\n");
      printGrid(rows, columns, h_life);

      for (int i = 1; i < processes; i++) {
         MPI_Recv(temp, rows * columns, MPI_CHAR, i, i, cartesian2D, &statusTemp);
         printf("The grid for process %d is:\n", i);
         printGrid(rows, columns, temp);
      }
      free(temp);
   }
   else {
      MPI_Send(d_life, rows * columns, MPI_CHAR, 0, rank, cartesian2D);
   }
#else
   if (rank == 0) {
      int rows_temp;
      for (int i = 1; i < processes; i++) {
         MPI_Recv(&rows_temp, 1, MPI_INT, i, i, cartesian2D, &statusTemp);
         printf("Process %d has %d rows\n", i, rows_temp);
      }
   }
   else {
      MPI_Send(&rows, 1, MPI_INT, 0, rank, cartesian2D);
   }
#endif
#endif
   for (int generation = 0; generation < GENERATIONS; generation++) {
      if (generation % 2 == 0) {
         recv = &receive_requests_even[0];
         send = &send_requests_even[0];
      } else {
         recv = &receive_requests_odd[0];
         send = &send_requests_odd[0];
      }


      MPI_Start(&send[4]);
      MPI_Start(&send[5]);
      MPI_Start(&send[6]);
      MPI_Start(&send[7]);
      
      nextGenerationInner(d_life, d_life_copy, rows, columns);



      nextGenerationOuter(d_life, d_life_copy, rows, columns);


      swap(&d_life, &d_life_copy);

      MPI_Waitall(4, send, statuses);
   }

   MPI_Pcontrol(0);
   t2 = MPI_Wtime();

   if (rank == 0) {
      printf("Elapsed time is %fs\n", (t2 - t1) );
   }

   // Clean up and exit
   free(h_life);
   cudaFree(d_life);
   cudaFree(d_life_copy);
   MPI_Type_free(&row_datatype);
   MPI_Type_free(&column_datatype);
   MPI_Finalize();
   return 0;
}


void inline
initialState(int rows, int columns, char *first_generation, int seed, MPI_Comm *cartesian2D, int rank) {
   float probability;
   srand(seed);

   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
         // Initialize all halo values to 0. The rest will be assigned values randomly
         if (i == 0 || j == 0 || i == rows - 1 || j == columns - 1) {
            *(first_generation + i * columns + j) = 0;
            continue;
         }
         probability = (float)rand() / (float)((unsigned)RAND_MAX + 1);
         if (probability >= 0.5f)
            *(first_generation + i * columns + j) = 1;
         else
            *(first_generation + i * columns + j) = 0;
      }
   }
}


void inline 
printGrid(int rows, int columns, char *life) {
   for (int i = 0; i< rows; i++) {
      for (int j = 0; j < columns; j++) {
         printf("%d ", *(life + i * columns + j));
         if ( j == columns - 1)
            printf("\n");
      }
   }
   printf("\n");
}

void inline
swap(char **a, char **b)
{
    char *temp = *a;
    *a = *b;
    *b = temp;
}
