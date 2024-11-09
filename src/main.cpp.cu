/*
Author: Adarsh Thirugnanasambandam Sriuma
Class: ECE6122 (A)
Last Date Modified: Nov 11, 2024

Description:

Program starting point. This file process user inputs
manages SFML windows and calls processes depending
on the input parameters.
*/

#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 600;
int CELL_SIZE = 5;
int PROCESS = 0; // 0 : NORMAL; 1 : PINNED; 2 : MANAGED
int NUMBER_OF_THREADS = 8;
int ROWS, COLS;

std::vector<std::vector<std::vector<int>>> quotas;
std::default_random_engine rng;
std::uniform_int_distribution<int> bw(0, 1);

struct Report
{
    int threadCount;
    std::string processName;
};

__global__ void gameOfLifeEngine(bool* input, bool* output, int* rows, int* cols)
{
    // Each thread gets one cell to compute
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread ID
    int r = tid / *cols; // current row idx
    int c = tid % *cols; // current column idx

    int cc = r * (*cols) + c; // 2D to 1D idx translation

    int tl_r, tm_r, tr_r, ml_r, mr_r, bl_r, bm_r, br_r;
    int tl_c, tm_c, tr_c, ml_c, mr_c, bl_c, bm_c, br_c;
    int tl, tm, tr, ml, mr, bl, bm, br;

    tl_r = r - 1; // top left row idx
    tm_r = r - 1; // top middle row idx
    tr_r = r - 1; // top right row idx
    ml_r = r;     // middle left row idx
    mr_r = r;     // middle RIGHT row idx
    bl_r = r + 1; // bottom left row idx
    bm_r = r + 1; // bottom middle row idx 
    br_r = r + 1; // bottom right row idx

    tl_c = c - 1; // same order as above but for columns
    tm_c = c;
    tr_c = c + 1;
    ml_c = c - 1;
    mr_c = c + 1;
    bl_c = c - 1;
    bm_c = c;
    br_c = c + 1;

    tl = input[tl_r * (*cols) + tl_c]; // corresponding cell value after 2D to 1D idx transposing
    tm = input[tm_r * (*cols) + tm_c];
    tr = input[tr_r * (*cols) + tr_c];
    ml = input[ml_r * (*cols) + ml_c];
    mr = input[mr_r * (*cols) + mr_c];
    bl = input[bl_r * (*cols) + bl_c];
    bm = input[bm_r * (*cols) + bm_c];
    br = input[br_r * (*cols) + br_c];

    int nearby_score = tl + tm + tr + ml + mr + bl + bm + br;
    
    if (input[cc] == 0)
    {
        if( nearby_score == 3)
            output[cc] = 1;
        else
            output[cc] = 0;
    }

    // alive cell: check death by isolation, death by overcrowding, or survival
    else 
    {
        // stay alive
        if (nearby_score == 2 | nearby_score == 3)
            output[cc] = 1;
        
        else
            output[cc] = 0;
    }

}

int main(int argc, char* argv[])
{
    rng.seed(time(NULL));
    for (int i=0; i< argc; i++)
    // process input args
    {
        std::string arg = argv[i];

        if (arg == "-n")
        {
            if (i+1 < argc)
                NUMBER_OF_THREADS = std::stoi(argv[i+1]);
        }
        else if (arg == "-c")
        {
            if (i+1 < argc)
                CELL_SIZE = std::stoi(argv[i+1]);
        }
        else if (arg == "-x")
        {
            if (i+1 < argc)
                WINDOW_WIDTH = std::stoi(argv[i+1]);
        }
        else if (arg == "-y")
        {
            if (i+1 < argc)
                WINDOW_HEIGHT = std::stoi(argv[i+1]);
        }
        else if (arg == "-t")
        {
            if (i+1 < argc)
            {
                if (std::string(argv[i+1]) == "PINNED")
                {
                    PROCESS = 1;
                }
                else if (std::string(argv[i+1]) == "MANAGED")
                {
                    PROCESS = 2;
                }
                else
                {
                    PROCESS = 0; // NORMAL
                }
            }
        }
    }

    // Report 
    Report performance;
    switch (PROCESS)
    {
    case 0:
        performance.threadCount = 1;
        performance.processName = " NORMAL ";
        break;
    
    case 1:
        performance.threadCount = NUMBER_OF_THREADS;
        performance.processName = " PINNED ";
        break;
    
    case 2:
        performance.threadCount = NUMBER_OF_THREADS;
        performance.processName = " MANAGED ";
        break;
    
    default:
        break;
    }

    // create the window
    sf::Texture texture;
    if (!texture.loadFromFile("white.png", sf::IntRect(0, 0, CELL_SIZE, CELL_SIZE)))
    {
        std::cout << "No texture found!";
    }
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "My window");
    
    COLS = (WINDOW_WIDTH-CELL_SIZE) / CELL_SIZE;
    ROWS = (WINDOW_HEIGHT-CELL_SIZE) / CELL_SIZE;

    std::vector<std::vector<sf::Sprite>> cells;
    for (int iRow=0; iRow < ROWS; iRow++)
    {
        std::vector<sf::Sprite> cells_row;
        for (int iCol=0; iCol < COLS; iCol++)
        {
            sf::Sprite cell(texture);
            cell.setPosition(iCol*CELL_SIZE, iRow*CELL_SIZE);
            cells_row.push_back(cell);
        }
        cells.push_back(cells_row);
    }

    // Buffer 1 (Active)
    bool** cgl_grid, **out_buff;
    switch (PROCESS)
    {
    case 0:
        cgl_grid = (bool**)malloc((ROWS + 2) * sizeof(bool*));
        cgl_grid[0] = (bool*)malloc((ROWS + 2) * (COLS + 2) * sizeof(bool));
        break;
    case 1:
        cudaMallocHost((void**)&cgl_grid, (ROWS + 2) * sizeof(bool*));
        cudaMallocHost((void**)&cgl_grid[0], (ROWS + 2) * (COLS + 2) * sizeof(bool));
        break;
    case 2:
        cudaMallocManaged(&cgl_grid, (ROWS + 2) * sizeof(bool*));
        cudaMallocManaged(&cgl_grid[0], (ROWS + 2) * (COLS + 2) * sizeof(bool));
        cudaMallocManaged(&out_buff, (ROWS + 2) * sizeof(bool*));
        cudaMallocManaged(&out_buff[0], (ROWS + 2) * (COLS + 2) * sizeof(bool));
    default:
        break;
    }
    
    for (int i = 1; i < ROWS + 2; i++) 
    {
        cgl_grid[i] = cgl_grid[i - 1] + (COLS + 2);
    }
    for (int i = 0; i < ROWS + 2; i++)
    {
        for (int j = 0; j < COLS + 2; j++)
        {
            cgl_grid[i][j] = bw(rng);
        }
    }
    for (int i = 0; i < COLS+2; i++) 
    {
        cgl_grid[0][i] = 0;
        cgl_grid[ROWS+1][i] = 0;
    }
    for (int i = 0; i < ROWS+2; i++)
    {
        cgl_grid[i][0] = 0;
        cgl_grid[i][COLS+1] = 0;
    }

    // Buffer 2 (Active)
    bool** cgl_grid_next;
    switch (PROCESS)
    {
    case 0:
        cgl_grid_next = (bool**)malloc((ROWS + 2) * sizeof(bool*));
        cgl_grid_next[0] = (bool*)malloc((ROWS + 2) * (COLS + 2) * sizeof(bool));
        break;
    case 1:
        cudaMallocHost((void**)&cgl_grid_next, (ROWS + 2) * sizeof(bool*));
        cudaMallocHost((void**)&cgl_grid_next[0], (ROWS + 2) * (COLS + 2) * sizeof(bool));
    case 2:
        cudaMallocManaged(&cgl_grid_next, (ROWS + 2) * sizeof(bool*));
        cudaMallocManaged(&cgl_grid_next[0], (ROWS + 2) * (COLS + 2) * sizeof(bool));
    default:
        break;
    }
    for (int i = 1; i < ROWS + 2; i++) 
    {
        cgl_grid_next[i] = cgl_grid_next[i - 1] + (COLS + 2);
    }
    for (int i = 0; i < ROWS + 2; i++)
    {
        for (int j = 0; j < COLS + 2; j++)
        {
            cgl_grid_next[i][j] = 0;
        }
    }

    bool *d_cgl_grid, *d_cgl_grid_next;
    if (PROCESS == 0 || PROCESS == 1)
    {
        cudaMalloc((void**)&d_cgl_grid, sizeof(bool) * (ROWS + 2) * (COLS + 2));
        cudaMalloc((void**)&d_cgl_grid_next, sizeof(bool) * (ROWS + 2) * (COLS + 2));

        cudaMemcpy(d_cgl_grid, cgl_grid[0], sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cgl_grid_next, cgl_grid[0], sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyHostToDevice);
    }
    // Pass number of rows and cols to GPU
    int *d_ROWS, *d_COLS;
    int rowsWithPadding, colsWithPadding;
    rowsWithPadding = ROWS + 2;
    colsWithPadding = COLS + 2;
    //printf("Rowswp: %d, Colswp: %d\n", rowsWithPadding, colsWithPadding);

    cudaMalloc((void**)&d_ROWS, sizeof(int));
    cudaMalloc((void**)&d_COLS, sizeof(int));
    cudaMemcpy(d_ROWS, &rowsWithPadding, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COLS, &colsWithPadding, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize = ceil(rowsWithPadding * colsWithPadding / NUMBER_OF_THREADS);

    sf::Clock Clock;
    Clock.restart();

    int generation_counter = 0;
    int Time = 0;
    int swap = 0;
    bool** display_buffer = cgl_grid_next;
    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
            
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Escape)
                {
                    std::cout << "ESCAPE key pressed" << std::endl;
                    window.close();
                    return 0;
                }
            }
        }

        if(generation_counter == 100)
        {
            generation_counter = 0;
            std::cout << "100 generations took " << Time << " microsecs " \
                << "with "<< performance.threadCount << " threads per block using " \
                << performance.processName << " memory allocation." << std::endl;
            Time = 0;
        }

        Clock.restart();
        if (!(swap % 2))
        {
            if (PROCESS == 0 || PROCESS == 1)
            {
                gameOfLifeEngine<<<blockSize, (dim3)NUMBER_OF_THREADS>>>(d_cgl_grid, d_cgl_grid_next, d_ROWS, d_COLS);
                cudaDeviceSynchronize();
                cudaMemcpy(cgl_grid_next[0], d_cgl_grid_next, sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyDeviceToHost);
                display_buffer = cgl_grid_next;
            }
            
            if (PROCESS == 2)
            {
                gameOfLifeEngine<<<blockSize, (dim3)NUMBER_OF_THREADS>>>(cgl_grid[0], cgl_grid_next[0], d_ROWS, d_COLS);
                display_buffer = cgl_grid_next;
                cudaDeviceSynchronize();
            }
        }
        else
        {
            if (PROCESS == 0 || PROCESS == 1)
            {
                gameOfLifeEngine<<<blockSize, (dim3)NUMBER_OF_THREADS>>>(d_cgl_grid_next, d_cgl_grid, d_ROWS, d_COLS);
                cudaDeviceSynchronize();
                cudaMemcpy(cgl_grid[0], d_cgl_grid, sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyDeviceToHost);
                display_buffer = cgl_grid;
            }
            if (PROCESS == 2)
            {
                gameOfLifeEngine<<<blockSize, (dim3)NUMBER_OF_THREADS>>>(cgl_grid_next[0], cgl_grid[0], d_ROWS, d_COLS);
                display_buffer = cgl_grid;
            }
        }


        Time += Clock.getElapsedTime().asMicroseconds();
        generation_counter++;

        // clear the window with black color
        window.clear(sf::Color::Black);
        for (int i_row = 1; i_row < ROWS-1; i_row++)
        {
            for (int i_col = 1; i_col < COLS-1; i_col++)
            {
                if (display_buffer[i_row][i_col])
                    window.draw(cells[i_row-1][i_col-1]);
            }
        }
        swap++;
        // sleep(1);

        // end the current frame
        window.display();
    }
    switch (PROCESS)
    {
    case 0:
        free(cgl_grid);
        free(cgl_grid_next);
        break;
    case 1:
        cudaFree(cgl_grid);
        cudaFree(cgl_grid_next);
        break;
    case 2:
        cudaFree(cgl_grid);
        cudaFree(cgl_grid_next);
        break;
    default:
        break;
    }

    return 0;
}
