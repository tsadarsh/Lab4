/*
Author: Adarsh Thirugnanasambandam Sriuma
Class: ECE6122 (A)
Last Date Modified: Oct 7, 2024

Description:

Program starting point. This file process user inputs
manages SFML windows and calls processes depending
on the input parameters.
*/

#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include "sequentialLogic.cpp"
#include "threadingLogic.cpp"
#include "OMPLogic.cpp"
#include <cuda.h>
#include <cuda_runtime.h>

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 600;
int CELL_SIZE = 5;
int PROCESS = 1; // 0 : SEQ; 1 : THRD; 2 : OMP
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

__global__ void vector_add(bool* d_cgl_grid, bool* d_cgl_grid_next, int* rows, int* cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = tid / *cols;
    int c = tid % *cols;

    int cc = r * (*cols) + c;

    int tl_r, tm_r, tr_r, ml_r, mr_r, bl_r, bm_r, br_r;
    int tl_c, tm_c, tr_c, ml_c, mr_c, bl_c, bm_c, br_c;
    int tl, tm, tr, ml, mr, bl, bm, br;

    tl_r = r - 1; 
    tm_r = r - 1;
    tr_r = r - 1;
    ml_r = r;
    mr_r = r;
    bl_r = r + 1;
    bm_r = r + 1;
    br_r = r + 1;

    tl_c = c - 1; 
    tm_c = c;
    tr_c = c + 1;
    ml_c = c - 1;
    mr_c = c + 1;
    bl_c = c - 1;
    bm_c = c;
    br_c = c + 1;

    tl = d_cgl_grid[tl_r * (*cols) + tl_c];
    tm = d_cgl_grid[tm_r * (*cols) + tm_c];
    tr = d_cgl_grid[tr_r * (*cols) + tr_c];
    ml = d_cgl_grid[ml_r * (*cols) + ml_c];
    mr = d_cgl_grid[mr_r * (*cols) + mr_c];
    bl = d_cgl_grid[bl_r * (*cols) + bl_c];
    bm = d_cgl_grid[bm_r * (*cols) + bm_c];
    br = d_cgl_grid[br_r * (*cols) + br_c];

    int nearby_score = tl + tm + tr + ml + mr + bl + bm + br;
    
    if (d_cgl_grid[cc] == 0)
    {
        if( nearby_score == 3)
            d_cgl_grid_next[cc] = 1;
    }

    // alive cell: check death by isolation, death by overcrowding, or survival
    else 
    {
        // death by isolation
        if (nearby_score <= 1)
            d_cgl_grid_next[cc] = 0;
        
        // death by overpopulation
        else if ( nearby_score >= 4)
            d_cgl_grid_next[cc] = 0;
        
        else{
            // cell survives, so do nothing
            d_cgl_grid_next[cc] = 1;
        }
    }

}

int main(int argc, char* argv[])
{
    rng.seed(time(NULL));
    for (int i=0; i< argc; i++)
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
                if (std::string(argv[i+1]) == "THRD")
                {
                    PROCESS = 1;
                }
                else if (std::string(argv[i+1]) == "OMP")
                {
                    PROCESS = 2;
                }
                else
                {
                    PROCESS = 0; // SEQ
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
        performance.processName = " sequential thread ";
        break;
    
    case 1:
        performance.threadCount = NUMBER_OF_THREADS;
        performance.processName = " std::threads ";
        break;
    
    case 2:
        performance.threadCount = NUMBER_OF_THREADS;
        performance.processName = " OMP threads ";
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

    // std::vector<std::vector<int>> cgl_grid;
    // std::vector<std::vector<int>> cgl_grid_next;

    bool** cgl_grid = (bool**)malloc((ROWS + 2) * sizeof(bool*));
    cgl_grid[0] = (bool*)malloc((ROWS + 2) * (COLS + 2) * sizeof(bool));
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


    bool** cgl_grid_next = (bool**)malloc((ROWS + 2) * sizeof(bool*));
    cgl_grid_next[0] = (bool*)malloc((ROWS + 2) * (COLS + 2) * sizeof(bool));
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
    cudaMalloc((void**)&d_cgl_grid, sizeof(bool) * (ROWS + 2) * (COLS + 2));
    cudaMalloc((void**)&d_cgl_grid_next, sizeof(bool) * (ROWS + 2) * (COLS + 2));

    cudaMemcpy(d_cgl_grid, cgl_grid[0], sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cgl_grid_next, cgl_grid[0], sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyHostToDevice);

    // Pass number of rows and cols to GPU
    int *d_ROWS, *d_COLS;
    int rowsWithPadding, colsWithPadding;
    rowsWithPadding = ROWS + 2;
    colsWithPadding = COLS + 2;
    printf("Rowswp: %d, Colswp: %d\n", rowsWithPadding, colsWithPadding);

    cudaMalloc((void**)&d_ROWS, sizeof(int));
    cudaMalloc((void**)&d_COLS, sizeof(int));
    cudaMemcpy(d_ROWS, &rowsWithPadding, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COLS, &colsWithPadding, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize = ceil(rowsWithPadding * colsWithPadding / NUMBER_OF_THREADS);
    printf("Blocksize: %d\n", blockSize);

    sf::Clock Clock;
    Clock.restart();

    int generation_counter = 0;
    int Time = 0;
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
            std::cout << "100 generations took " << Time << " microseconds " \
                << "with "<< performance.threadCount << performance.processName << std::endl;
            Time = 0;
        }

        Clock.restart();
        vector_add<<<blockSize, (dim3)NUMBER_OF_THREADS>>>(d_cgl_grid, d_cgl_grid_next, d_ROWS, d_COLS);
        cudaMemcpy(cgl_grid_next[0], d_cgl_grid_next, sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_cgl_grid, cgl_grid_next[0], sizeof(bool) * (ROWS + 2) * (COLS + 2), cudaMemcpyHostToDevice);

        // switch (PROCESS)
        // {
        // case 0:
        //     SEQ_Process(&cgl_grid, &cgl_grid_next);
        //     break;
        
        // case 1:
        //     // THRD Process
        //     THRD_Process(&cgl_grid, &cgl_grid_next, NUMBER_OF_THREADS, &quotas);
        //     break;

        // case 2:
        //     // OMP Process
        //     OMP_Process(&cgl_grid, &cgl_grid_next, NUMBER_OF_THREADS);
        //     break;
        
        // default:
        //     break;
        // }
        Time += Clock.getElapsedTime().asMicroseconds();
        generation_counter++;

        // clear the window with black color
        window.clear(sf::Color::Black);
        for (int i_row = 1; i_row < ROWS-1; i_row++)
        {
            for (int i_col = 1; i_col < COLS-1; i_col++)
            {
                if (cgl_grid_next[i_row][i_col])
                    window.draw(cells[i_row-1][i_col-1]);
            }
        }

        // end the current frame
        window.display();
    }

    return 0;
}