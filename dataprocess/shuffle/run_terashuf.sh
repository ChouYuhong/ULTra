g++ -std=c++11 -O2 -o terashuf terashuf.cpp
cat *.jsonl TMPDIR=/path/to/tmp MEMORY=20 | ./terashuf | split -l 100000 -a 5 -d - output/shuffled-
