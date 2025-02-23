# Numerical-Example-for-a-Simple-AnnN
import numpy as np

# دالة  السيجمويد
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(inputs, weights, biases):
    net_hidden = np.dot(weights["weights_input_hidden"], inputs) + biases["bias_hidden"]
    output_hidden = sigmoid(net_hidden)
    
    net_output = np.dot(weights["weights_hidden_output"], output_hidden) + biases["bias_output"]
    the_final_output = sigmoid(net_output)
    
    return output_hidden, the_final_output

def squared_error(target, output):
    return 0.5 * np.square(target - output)

# القيم المدخلة المطلوبة
input_values = np.array([0.1, 0.2, 1])  #  bias2 غير قيمه bias1  لان قيمه i1, i2، وقيمة الـ bias 

target_values = np.array([0.1, 0.99])

# الأوزان والقيم المستخدمة في المسئله
weights = {
    "weights_input_hidden": np.array([[0.15, 0.2, 0.35],
                                       [0.25, 0.3, 0.35]]),
    "weights_hidden_output": np.array([[0.4, 0.45],
                                         [0.5, 0.55]])
}

biases = {
    "bias_hidden": np.array([0.35, 0.35]),
    "bias_output": np.array([0.6, 0.6])
}

hidden_layer_output, the_final_output = forward_pass(input_values, weights, biases)

#هنحسب هناالخطأ لكل مخرج
error_values = squared_error(target_values, the_final_output)

total_error_value = np.sum(error_values)

# طباعة النتائج النهائية
print("  ناتج الطبقة المخفية::", hidden_layer_output)
print("النتيجه النهاءيه:",the_final_output)
print("الايرور:", error_values)
print("إجمالي الخطأ:", total_error_value)
