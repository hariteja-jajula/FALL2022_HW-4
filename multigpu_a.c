#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define SIZE 840
#define NDIMS 2
#define GENERATIONS 1000
#define BLOCKS

void Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy, int seed);
void Print_grid(int rows, int columns, char *life);
void inline Next_generation_inner(int rows, int columns, char *life, char *life_copy);
void inline Next_generation_outer(int rows, int columns, char *life, char *life_copy);
void inline Swap(char **a, char **b);

int main()
{


    int dim_size[NDIMS], periods[NDIMS], coords[NDIMS];
    int north[NDIMS], east[NDIMS], south[NDIMS], west[NDIMS],
        northeast[NDIMS], southeast[NDIMS], southwest[NDIMS], northwest[NDIMS];

    int            reorder, rank, processes, rows, columns, seed;
    int            north_rank, east_rank, south_rank, west_rank,
                   northeast_rank, southeast_rank, southwest_rank, northwest_rank;
    MPI_Comm       cartesian2D;


    MPI_Datatype   row_datatype, column_datatype;
    MPI_Request    receive_requests_even[8], send_requests_even[8], receive_requests_odd[8], send_requests_odd[8];
    MPI_Status     statuses[8];
    double         t1, t2, root;

    periods[0] = periods[1] = 1;

    reorder = 1;

    /* Initialize MPI */
    MPI_Init(NULL, NULL);
    MPI_Pcontrol(0);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef BLOCKS
    root = sqrt((double)processes);

    if (root == floor(root))
        dim_size[0] = dim_size[1] = (int)root;
    else
        dim_size[0] = dim_size[1] = 0;
#else
    dim_size[0] = processes;
    dim_size[1] = 1;
#endif

    /* Let MPI decide which is the best arrangement according to the number of processes and dimensions */
    if ( MPI_Dims_create(processes, NDIMS, dim_size) != MPI_SUCCESS )
    {
        if (rank == 0)
            printf("Number of processes and size of grid do not match. MPI_Dims_create() returned an error. Exiting.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
        MPI_Finalize();
        return -1;
    }

    /* Create a 2D Cartesian topology. Find the rank and coordinates of each process */
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dim_size, periods, reorder, &cartesian2D);
    MPI_Cart_coords(cartesian2D, rank, NDIMS, coords);

#ifdef BLOCKS
    /* We add 2 to each dimension in order to include the halo rows and columns */
    rows = (SIZE / dim_size[0]) + 2;
    columns = (SIZE /dim_size[1]) + 2;
#else
    /* If the size of the grid divides evely by the number of processes, then every process gets the same amount of rows... */
    if (SIZE % processes == 0)
        rows = SIZE / processes;
    else
    {
        /* ...otherwise, the last process will get a few more rows */
        if (rank != processes - 1)
            rows = SIZE / processes;
        else
            rows = SIZE - (SIZE / processes) * (processes - 1);
    }
    rows += 2;
    columns = SIZE + 2;
#endif

    /* Calculate the coordinates and ranks of all neighbors */
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

    /* We need two datatypes for the halos, one for the rows and one for the columns */
    MPI_Type_contiguous(columns - 2, MPI_CHAR, &row_datatype);
    MPI_Type_commit(&row_datatype);

    MPI_Type_vector(rows - 2, 1, columns, MPI_CHAR, &column_datatype);
    MPI_Type_commit(&column_datatype);

    /* Pointers to our 2D grid, and its necessary copy */
    char *life = (char*)malloc( rows * columns * sizeof(char) );
    char *life_copy = (char*)malloc( rows * columns * sizeof(char) );

    /* Generate the first generation according to the random seed */
    seed = rank + 2;
    Initial_state(rows, columns, life, life_copy, seed);

    MPI_Recv_init( life + 1, 1, row_datatype, north_rank, north_rank, cartesian2D, &receive_requests_even[0] );
    MPI_Recv_init( life + (rows - 1) * columns + 1, 1, row_datatype, south_rank, south_rank, cartesian2D, &receive_requests_even[1] );
    MPI_Recv_init( life + columns, 1, column_datatype, west_rank, west_rank, cartesian2D, &receive_requests_even[2] );
    MPI_Recv_init( life + (columns * 2) - 1, 1, column_datatype, east_rank, east_rank, cartesian2D, &receive_requests_even[3] );

    MPI_Recv_init( life, 1, MPI_CHAR, northwest_rank, northwest_rank, cartesian2D, &receive_requests_even[4] );
    MPI_Recv_init( life + columns - 1, 1, MPI_CHAR, northeast_rank, northeast_rank, cartesian2D, &receive_requests_even[5] );
    MPI_Recv_init( life + columns * (rows - 1), 1, MPI_CHAR, southwest_rank, southwest_rank, cartesian2D, &receive_requests_even[6] );
    MPI_Recv_init( life + (columns * rows) - 1, 1, MPI_CHAR, southeast_rank, southeast_rank, cartesian2D, &receive_requests_even[7] );

    MPI_Send_init( life + (rows - 2) * columns + 1, 1, row_datatype, south_rank, rank, cartesian2D, &send_requests_even[0] );
    MPI_Send_init( life + columns + 1, 1, row_datatype, north_rank, rank, cartesian2D, &send_requests_even[1] );
    MPI_Send_init( life + (columns * 2) - 2, 1, column_datatype, east_rank, rank, cartesian2D, &send_requests_even[2] );
    MPI_Send_init( life + columns + 1, 1, column_datatype, west_rank, rank, cartesian2D, &send_requests_even[3] );

    MPI_Send_init( life + columns * (rows - 1) - 2, 1, MPI_CHAR, southeast_rank, rank, cartesian2D, &send_requests_even[4] );
    MPI_Send_init( life + columns * (rows - 2) + 1, 1, MPI_CHAR, southwest_rank, rank, cartesian2D, &send_requests_even[5] );
    MPI_Send_init( life + (columns * 2) - 2, 1, MPI_CHAR, northeast_rank, rank, cartesian2D, &send_requests_even[6] );
    MPI_Send_init( life + columns + 1, 1, MPI_CHAR, northwest_rank, rank, cartesian2D, &send_requests_even[7] );

    /**************************************************************************************/
    /* These are for the odd iterations of the loop, e.g. generation = 1, 3, 5, 7, 9 etc. */
    /**************************************************************************************/
    MPI_Recv_init( life_copy + 1, 1, row_datatype, north_rank, north_rank, cartesian2D, &receive_requests_odd[0] );
    MPI_Recv_init( life_copy + (rows - 1) * columns + 1, 1, row_datatype, south_rank, south_rank, cartesian2D, &receive_requests_odd[1] );
    MPI_Recv_init( life_copy + columns, 1, column_datatype, west_rank, west_rank, cartesian2D, &receive_requests_odd[2]) ;
    MPI_Recv_init( life_copy + (columns * 2) - 1, 1, column_datatype, east_rank, east_rank, cartesian2D, &receive_requests_odd[3] );

    MPI_Recv_init( life_copy, 1, MPI_CHAR, northwest_rank, northwest_rank, cartesian2D, &receive_requests_odd[4] );
    MPI_Recv_init( life_copy + columns - 1, 1, MPI_CHAR, northeast_rank, northeast_rank, cartesian2D, &receive_requests_odd[5] );
    MPI_Recv_init( life_copy + columns * (rows - 1), 1, MPI_CHAR, southwest_rank, southwest_rank, cartesian2D, &receive_requests_odd[6] );
    MPI_Recv_init( life_copy + (columns * rows) - 1, 1, MPI_CHAR, southeast_rank, southeast_rank, cartesian2D, &receive_requests_odd[7] );

    MPI_Send_init( life_copy + (rows - 2) * columns + 1, 1, row_datatype, south_rank, rank, cartesian2D, &send_requests_odd[0] );
    MPI_Send_init( life_copy + columns + 1, 1, row_datatype, north_rank, rank, cartesian2D, &send_requests_odd[1] );
    MPI_Send_init( life_copy + (columns * 2) - 2, 1, column_datatype, east_rank, rank, cartesian2D, &send_requests_odd[2] );
    MPI_Send_init( life_copy + columns + 1, 1, column_datatype, west_rank, rank, cartesian2D, &send_requests_odd[3] );

    MPI_Send_init( life_copy + columns * (rows - 1) - 2, 1, MPI_CHAR, southeast_rank, rank, cartesian2D, &send_requests_odd[4] );
    MPI_Send_init( life_copy + columns * (rows - 2) + 1, 1, MPI_CHAR, southwest_rank, rank, cartesian2D, &send_requests_odd[5] );
    MPI_Send_init( life_copy + (columns * 2) - 2, 1, MPI_CHAR, northeast_rank, rank, cartesian2D, &send_requests_odd[6] );
    MPI_Send_init( life_copy + columns + 1, 1, MPI_CHAR, northwest_rank, rank, cartesian2D, &send_requests_odd[7] );

    /* Synchronize all the processes before we start */
    MPI_Barrier(cartesian2D);
    t1 = MPI_Wtime();
    MPI_Pcontrol(1);

#ifdef DEBUG_GRID
#ifdef BLOCKS
    /* Print the grid of every process, before the exchange of the halo elements and before the beginning of the main loop*/
    if (rank == 0)
    {
        MPI_Status status;
        char* process2 = (char *)malloc( rows * columns * sizeof(char));

        printf("The grid for process 0 is:\n");
        Print_grid(rows, columns, life);

        for (int i = 1; i < processes; i++)
        {
            MPI_Recv(process2, rows * columns, MPI_CHAR, i, i, cartesian2D, &status);
            printf("The grid for process %d is:\n", i);
            Print_grid(rows, columns, process2);

        }
        free(process2);
    }
    else
        MPI_Send(life, rows * columns, MPI_CHAR, 0, rank, cartesian2D);
#else
    if (rank == 0)
    {
        MPI_Status status;
        int rows_temp;
        for (int i = 1; i < processes; i++)
        {
            MPI_Recv(&rows_temp, 1, MPI_INT, i, i, cartesian2D, &status);
            printf("Process %d has %d rows\n", i, rows_temp);
        }
    }
    else
    {
        MPI_Send(&rows, 1, MPI_INT, 0, rank, cartesian2D);
    }
#endif
#endif

    /* Modify the number of generations as desired */
    for (int generation = 0; generation < GENERATIONS; generation++)
    {
        if (generation % 2 == 0)
        {
            MPI_Start(&receive_requests_even[0]);
            MPI_Start(&receive_requests_even[1]);
            MPI_Start(&receive_requests_even[2]);
            MPI_Start(&receive_requests_even[3]);
            MPI_Start(&receive_requests_even[4]);
            MPI_Start(&receive_requests_even[5]);
            MPI_Start(&receive_requests_even[6]);
            MPI_Start(&receive_requests_even[7]);

            MPI_Start(&send_requests_even[0]);
            MPI_Start(&send_requests_even[1]);
            MPI_Start(&send_requests_even[2]);
            MPI_Start(&send_requests_even[3]);
            MPI_Start(&send_requests_even[4]);
            MPI_Start(&send_requests_even[5]);
            MPI_Start(&send_requests_even[6]);
            MPI_Start(&send_requests_even[7]);

            Next_generation_inner(rows, columns, life, life_copy);

            MPI_Waitall(8, receive_requests_even, statuses);

            Next_generation_outer(rows, columns, life, life_copy);
            
#ifdef DEBUG_GRID
#ifdef BLOCKS
            /* Print the grid of every process */
            if (rank == 0)
            {
                printf("Generation %d:\n", generation);
                MPI_Status status;
                char* process2 = (char*)malloc(rows * columns * sizeof(char));
                printf("The grid for process 0 is:\n");
                Print_grid(rows, columns, life);

                for (int i = 1; i < processes; i++)
                {
                    MPI_Recv(process2, rows * columns, MPI_CHAR, i, i, cartesian2D, &status);
                    printf("The grid for process %d is:\n", i);
                    Print_grid(rows, columns, process2);
                }
                free(process2);
            }
            else
                MPI_Send(life, rows * columns, MPI_CHAR, 0, rank, cartesian2D);
#endif
#endif

            Swap(&life, &life_copy);
            MPI_Waitall(8, send_requests_even, statuses);
        }
        else
        {
            MPI_Start(&receive_requests_odd[0]);
            MPI_Start(&receive_requests_odd[1]);
            MPI_Start(&receive_requests_odd[2]);
            MPI_Start(&receive_requests_odd[3]);
            MPI_Start(&receive_requests_odd[4]);
            MPI_Start(&receive_requests_odd[5]);
            MPI_Start(&receive_requests_odd[6]);
            MPI_Start(&receive_requests_odd[7]);

            MPI_Start(&send_requests_odd[0]);
            MPI_Start(&send_requests_odd[1]);
            MPI_Start(&send_requests_odd[2]);
            MPI_Start(&send_requests_odd[3]);
            MPI_Start(&send_requests_odd[4]);
            MPI_Start(&send_requests_odd[5]);
            MPI_Start(&send_requests_odd[6]);
            MPI_Start(&send_requests_odd[7]);

            Next_generation_inner(rows, columns, life, life_copy);

            MPI_Waitall(8, receive_requests_odd, statuses);

            Next_generation_outer(rows, columns, life, life_copy);

#ifdef DEBUG_GRID
#ifdef BLOCKS

            if (rank == 0)
            {
                printf("Generation %d:\n", generation);
                MPI_Status status;
                char* process2 = (char*)malloc(rows * columns * sizeof(char));
                printf("The grid for process 0 is:\n");
                Print_grid(rows, columns, life);

                for (int i = 1; i < processes; i++)
                {
                    MPI_Recv(process2, rows * columns, MPI_CHAR, i, i, cartesian2D, &status);
                    printf("The grid for process %d is:\n", i);
                    Print_grid(rows, columns, process2);

                }
                free(process2);
            }
            else
                MPI_Send(life, rows * columns, MPI_CHAR, 0, rank, cartesian2D);
#endif
#endif

            Swap(&life, &life_copy);
            MPI_Waitall(8, send_requests_odd, statuses);
        }
    }

    MPI_Pcontrol(0);
    t2 = MPI_Wtime();

    if (rank == 0)
        printf("Elapsed time is %fs\n", (t2 - t1) );

    /* Clean up and exit */
    free(life);
    free(life_copy);
    MPI_Type_free(&row_datatype);
    MPI_Type_free(&column_datatype);
    MPI_Finalize();
    return 0;
}

void inline Initial_state(int rows, int columns, char *first_generation, char *first_generation_copy, int seed)
{
    float probability;
    srand(seed);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            /* Initialize all halo values to 0. The rest will be assigned values randomly */
            if ( i == 0 || j == 0 || i == rows - 1 || j == columns - 1)
            {
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 0;
                continue;
            }
            probability = (float)rand() / (float)((unsigned)RAND_MAX + 1);
            if (probability >= 0.5f)
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 1;
            else
                *(first_generation + i * columns + j) = *(first_generation_copy + i * columns + j) = 0;
        }
    }
}


void inline Print_grid(int rows, int columns, char *life)
{
    for (int i = 0; i< rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            printf("%d ", *(life + i * columns + j));
            if ( j == columns - 1)
                printf("\n");
        }
    }
    printf("\n");
}


void inline Next_generation_inner(int rows, int columns, char *life, char *life_copy)
{
    int neighbors;
    for (int i = 2; i < rows - 2; i++)
    {
        for (int j = 2; j < columns - 2; j++)
        {
            neighbors = *(life + (i - 1) * columns + (j - 1)) + *(life + (i - 1) * columns + j) + *(life + (i - 1) * columns + (j + 1)) +
                        *(life + i * columns + (j - 1))                          +                *(life + i * columns + (j + 1))       +
                        *(life + (i + 1) * columns + (j - 1)) + *(life + (i + 1) * columns + j) + *(life + (i + 1) * columns + (j + 1));

            if (neighbors == 3 || (neighbors == 2 && *(life_copy + i * columns + j) == 1))
                *(life_copy + i * columns + j) = 1;
            else
                *(life_copy + i * columns + j) = 0;
        }
    }
}

void inline Next_generation_outer(int rows, int columns, char *life, char *life_copy)
{
    int neighbors;

    /* Upper row */
    for (int i = 1; i < columns - 1; i++)
    {
        neighbors = *(life + i - 1)               + *(life + i)  +              *(life + i + 1)               +
                    *(life + columns + i - 1)     + /* you are here */          *(life + columns + i + 1)     +
                    *(life + columns * 2 + i - 1) + *(life + columns * 2 + i) + *(life + columns * 2 + i + 1);

        if (neighbors == 3 || (neighbors == 2 && *(life_copy + columns + i) == 1))
            *(life_copy + columns + i) = 1;
        else
            *(life_copy + columns + i) = 0;
    }

    /* Left column */
    for (int i = 2; i < rows - 2; i++)
    {
        neighbors = *(life + columns * (i - 1)) + *(life + columns * (i - 1) + 1) + *(life + columns * (i - 1) + 2) +
                    *(life + columns * i)       + /* you are here */                *(life + columns * i + 2)       +
                    *(life + columns * (i + 1)) + *(life + columns * (i + 1) + 1) + *(life + columns * (i + 1) + 2);

        if (neighbors == 3 || (neighbors == 2 && *(life_copy + columns * i + 1) == 1))
            *(life_copy + columns * i + 1) = 1;
        else
            *(life_copy + columns * i + 1) = 0;   
    }

    /* Right column */
    for (int i = 2; i < rows - 2; i++)
    {
        neighbors = *(life + columns * i - 3)       + *(life + columns * i - 2)       + *(life + columns * i - 1)             +
                    *(life + columns * (i + 1) - 3) + /* you are here */                *(life + columns * (i + 1) - 1)       +
                    *(life + columns * (i + 2) - 3) + *(life + columns * (i + 2) - 2) + *(life + columns * (i + 2) - 1);

        if (neighbors == 3 || (neighbors == 2 && *(life_copy + columns * (i + 1) - 2) == 1))
            *(life_copy + columns * (i + 1) - 2) = 1;
        else
            *(life_copy + columns * (i + 1) - 2) = 0;
    }

    /* Bottom row */
    for (int i = 1; i < columns - 1; i++)
    {
        neighbors = *(life + columns * (rows - 3) + i - 1) + *(life + columns * (rows - 3) + i) + *(life + columns * (rows - 3) + i + 1)     +
                    *(life + columns * (rows - 2) + i - 1) + /* you are here */                   *(life + columns * (rows - 2) + i + 1)     +
                    *(life + columns * (rows - 1) + i - 1) + *(life + columns * (rows - 1) + i) + *(life + columns * (rows - 1) + i + 1);

        if (neighbors == 3 || (neighbors == 2 && *(life_copy + columns * (rows - 2) + i) == 1))
            *(life_copy + columns * (rows - 2) + i) = 1;
        else
            *(life_copy + columns * (rows - 2) + i) = 0;
    }
}

void inline Swap(char **a, char **b)
{
    char *temp = *a;
    *a = *b;
    *b = temp;
}
