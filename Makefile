task: task.cpp
	mpic++ $^ -o $@ -fopenmp -std=c++14 -Wall -Wextra

NODES := 2
AMOUNT := 20
run: task
	mpirun -n $(NODES) ./task $(AMOUNT)

clean:
	@$(RM) task
