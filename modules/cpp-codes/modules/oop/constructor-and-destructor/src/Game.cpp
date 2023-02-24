#include <iostream>
#include "Game.h"

void Game::purchase(const std::string &title, float value)
{
    m_name = title;
    m_price = value;
    m_hours = 0;
    m_cost = m_price;
}

void Game::update(float cost)
{
    m_price = cost;
    calculate();
}

void Game::play(int hours)
{
    hours += hours;
    calculate();
}

void Game::showInformation()
{
    std::cout << m_name << " R$"
              << m_price << " "
              << m_hours << "h = R$"
              << m_cost << "/h\n";
}