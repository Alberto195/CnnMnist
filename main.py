from model import *

(x_train, y_train), (x_test, y_test) = load_data()

epochs = set_epochs(10)

(x_train, y_train), (x_test, y_test), input_shape = data_stuff(x_train, y_train, x_test, y_test)

model = model_create(input_shape)

model_compile(model)

history = model_train(x_train, y_train, x_test, y_test, model, epochs)

model_eval(model, x_test, y_test)

draw_graph(history)
