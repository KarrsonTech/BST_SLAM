#include <Program.hpp>
#include <SensorTest.hpp>

int main() {
	if (VIO::Program::ShouldTestSensor) return (new VIO::SensorTest())->Main();
	else return (new VIO::Program())->Main();
}