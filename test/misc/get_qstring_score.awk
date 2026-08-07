#!/bin/awk -f

# Function to calculate the logarithm base 10
function log10(x) {
    return log(x) / log(10)
}

# Function to get the ASCII value of a character
function ord(c) {
    return index(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~", c) + 31
}

BEGIN {
    # Convert the qstring to an array of integers (qs) by subtracting 33 from each character's ASCII value
    for (i = 1; i <= length(qstring); i++) {
        qs[i] = ord(substr(qstring, i, 1)) - 33
    }
    
    # Calculate mean_err
    sum_exp = 0
    for (i = 1; i <= length(qs); i++) {
        sum_exp += exp(qs[i] * (-log(10) / 10.0))
    }
    mean_err = sum_exp / length(qs)
    
    # Calculate the score
    score = -10 * log10((mean_err > 1e-4) ? mean_err : 1e-4)
    
    # Print the score
    print score
}
