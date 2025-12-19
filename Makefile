CXX = g++
CXXFLAGS = -O3 -Wall
LDFLAGS = -lzstd

all: compressor

compressor: src/compressor.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o build/compressor src/compressor.cpp $(LDFLAGS)

clean:
	rm -f build/compressor
