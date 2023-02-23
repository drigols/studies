#include <string>

class Game
{
private:
    // Encapsulation.
    std::string m_name; // Game name.
    float m_price;      // Game price.
    int m_hours;        // Hours played.
    float m_cost;       // Cost per hour player.

    // Calculate the cost to played hours (Inline function/Method).
    void calculate()
    {
        if (m_hours > 0)
            m_cost = m_price / m_hours;
    }

public:
    // Interfaces.
    void purchase(const std::string &title, float value); // Fill the information.
    void update(float value);                             // Update game price.
    void play(int time);                                  // Record (save) the hours played.
    void showInformation();                               // show information.
};
