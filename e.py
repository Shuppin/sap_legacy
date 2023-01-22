
def find_num(num):
    num_stack = [1,2,3,4]
    pop_stack = []
    

    while len(num_stack) > 0:
        current_num = num_stack.pop()
        pop_stack.append(current_num)
        if current_num == num:
            while len(pop_stack) > 0:
                num_stack.append(pop_stack.pop())
            print("Found")
            return num
    
    while len(pop_stack) > 0:
        num_stack.append(pop_stack.pop())

    print("Not found")

find_num(6)