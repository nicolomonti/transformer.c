CC = gcc
CFLAGS = -Wall -g
LDFLAGS = -lm

TARGET = main

all: $(TARGET)

$(TARGET): main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(TARGET)
