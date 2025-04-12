CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src
LDFLAGS = 

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

TARGET = $(BIN_DIR)/main
TEST_TARGET = $(BIN_DIR)/tests

.PHONY: all clean test

all: $(TARGET)

test: $(TEST_TARGET)
	@./$(TEST_TARGET)

$(TARGET): $(filter-out $(OBJ_DIR)/tests.o,$(OBJS))
	@mkdir -p $(BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

$(TEST_TARGET): $(filter-out $(OBJ_DIR)/main.o,$(OBJS))
	@mkdir -p $(BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

