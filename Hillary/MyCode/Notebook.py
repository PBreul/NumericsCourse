def square(x):
    print(x, x**2)

func_dic={"square":square}

#func_dic["square"](2)
f=func_dic["square"]
f(2)