CC=g++
CXXFLAGS=-O3 `pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`


.PHONY: all clean
FRIF: main.cpp FrifDescriptor.o FrifDetector.o utils.h utils.cpp
	$(CC) $(CXXFLAGS) -o $@ FrifDescriptor.o FrifDetector.o utils.cpp main.cpp $(LDFLAGS)

FrifDescriptor.o: Common.h FrifDescriptor.h FrifDescriptor.cpp
	$(CC) $(CXXFLAGS) -c FrifDescriptor.cpp -o $@

FrifDetector.o: Common.h FrifDetector.h FrifDetector.cpp
	$(CC) $(CXXFLAGS) -c FrifDetector.cpp -o $@


all: FRIF

clean:
	rm -f *.o FRIF

