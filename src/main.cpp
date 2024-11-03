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

/**
 * createQuota - Batches current generations grid and stores the indices in `quotas` vector
 *
 * @num_of_threads: Number of threads that was requested to be spawned
 * 
 * @cgl_grid: Grid that holds the current generation status
 */

void createQuota(int num_of_threads, std::vector<std::vector<int>>* cgl_grid)
{
    std::vector<std::vector<int>> threadQuota;
    int batchSize = (ROWS * COLS) / NUMBER_OF_THREADS;

    for (int iRow = 1; iRow < (*cgl_grid).size()-1; iRow++)
    {
        for (int iCol = 1; iCol < (*cgl_grid)[0].size()-1; iCol++)
        {
            std::vector<int> rowColIdx;

            rowColIdx.push_back(iRow);
            rowColIdx.push_back(iCol);

            if (threadQuota.size() == batchSize)
            {
                quotas.push_back(std::move(threadQuota));
                threadQuota.resize(0);
            }

            threadQuota.push_back(std::move(rowColIdx));
        }
    }
    if (threadQuota.size() > 0)
    {
        quotas.push_back(std::move(threadQuota));
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

    std::vector<std::vector<int>> cgl_grid;
    std::vector<std::vector<int>> cgl_grid_next;
    // +2 for row and cols to have a boundry of 0s
    for (int i_row=0; i_row < ROWS+2; i_row++)
    {
        std::vector<int> cgl_grid_row;
        std::vector<int> cgl_grid_row_next;
        for (int i_col=0; i_col < COLS+2; i_col++)
        {
            int val = bw(rng);
            // std::cout << val;
            cgl_grid_row.push_back(val);
            cgl_grid_row_next.push_back(0);
        }
        cgl_grid.push_back(cgl_grid_row);
        cgl_grid_next.push_back(cgl_grid_row);
        // std::cout << std::endl;
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


    if (PROCESS == 1)
    {
        createQuota(NUMBER_OF_THREADS, &cgl_grid);
    }

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

        // Clock.restart();
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
        // Time += Clock.getElapsedTime().asMicroseconds();
        // generation_counter++;

        // clear the window with black color
        window.clear(sf::Color::Black);
        for (int i_row = 1; i_row < cgl_grid_next.size()-1; i_row++)
        {
            for (int i_col = 1; i_col < cgl_grid_next[i_row].size()-1; i_col++)
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