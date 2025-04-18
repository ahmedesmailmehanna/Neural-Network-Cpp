#ifndef TEST_RUNNER_HPP
#define TEST_RUNNER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

class TestRunner {
    private:
        int totalTests = 0;
        int passedTests = 0;
        std::string currentTestName;
    
        void testResult(bool passed) {
            totalTests++;
            if (passed) {
                passedTests++;
                std::cout << currentTestName << " PASSED" << std::endl;
            } else {
                std::cout << currentTestName << " FAILED" << std::endl;
            }
        }
    
    public:
        void runTest(const std::string name, std::function<bool()> test) {
            currentTestName = name;
            bool result = test();
    
            testResult(result);
        }
    
        void printSummary() {
            std::cout << "\nTest Summary:" << std::endl;
            std::cout << "Total Tests: " << totalTests << std::endl;
            std::cout << "Passed: " << passedTests << std::endl;
            std::cout << "Failed: " << (totalTests - passedTests) << std::endl;
            std::cout << "Success Rate: " << (static_cast<double>(passedTests) / totalTests * 100) << "%" << std::endl;
        }
    };

#endif  // TEST_RUNNER_HPP