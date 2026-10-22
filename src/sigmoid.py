import numpy as np

def sigmoid(z):
    """
      Convert a score into a probability between  and 1
    """
    return 1/(1+np.exp(-z))



if __name__=="__main__":
    test_values = np.array([-2,-1,0,1,2])

    probabilities = sigmoid(test_values)


    for z , probability in zip(test_values,probabilities):
        print(f"z ={z:2}| Probability = {probability:.4f}")
