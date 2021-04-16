from sklearn.model_selection import train_test_split

num = 10
idx_x = [x for x in range(num)]
x = [x+3 for x in range(num)]
y = [pow(x,2) for x in range(num)]
print(f"indx_x {idx_x}\n")
print(f"x {x}\n")
print(f"y {y}\n")
#x_train, x_test, y_train, y_test = train_test_split(x, test_size=0.1)
x_train, x_test = train_test_split(x, test_size=0.1)

print(f"x_train:{x_train}")
print(f"x_test:{x_test}")
#print(f"y_train:{y_train}")
#print(f"y_test:{y_test}")
