CC = gcc
CFLAGS = -O3 -lm
TARGET = net

all: $(TARGET)

$(TARGET): net.c
	$(CC) $^ $(CFLAGS) -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean

