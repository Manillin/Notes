#include <iostream>
#include <vector>

int main()
{
    float k = 0.5;
    float ambient_temp = 20;
    std::vector<float> temp = {42, 24, 50};

    auto op = [=](float t)
    {
        float diff = ambient_temp - t;
        return t + k * diff;
    };
    for (int step = 0; step < 3; step++)
    {
        std::transform(temp.begin(), temp.end(),
                       temp.begin(), op);
    }
}