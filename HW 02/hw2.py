import sys
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0]*26
    s = [0]*26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char)-ord('A')] = float(prob)
    f.close()

    return (e, s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    with open(filename, encoding='utf-8') as f:
        # TODO: add your code here
        for lst in f:
            # Read from the list
            for c in lst:
                c = c.upper()
                # check the letter if it is inside the boundary
                if 'A' <= c and c <= 'Z':
                    if c in X:
                        X[c] += 1
                    else:
                        X[c] = 1
    return X


# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

    
def main():
    print("Q1")

    X = shred('letter.txt')
    for i in range(ord('A'), ord('Z') + 1):
        c = chr(i)
        if c in X.keys():
            print("{} {}".format(c, X[c]))
        else:
            print("{} 0".format(c))

    print("Q2")

    tuple = get_parameter_vectors()
    if 'A' in X.keys():
        X1e1 = X['A'] * math.log(tuple[0][0])
        X1s1 = X['A'] * math.log(tuple[1][0])
        print("%.4f\n%.4f" % (X1e1, X1s1))
    else:
        print("-0.0000\n-0/0000")

    print("Q3")

    Feng = math.log(0.6)
    Fspa = math.log(0.4)

    for i in range(ord('A'), ord('Z') + 1):
        c = chr(i)
        if c in X.keys():
            Feng += X[c] * math.log(tuple[0][ord(c) - ord('A')])
            Fspa += X[c] * math.log(tuple[1][ord(c) - ord('A')])

    print("%.4f\n%.4f" % (Feng, Fspa))

    print("Q4")
    if (Fspa - Feng >= 100):
        print("0.0000")
    elif (Fspa - Feng <= -100):
        print("1.0000")
    else:
        print("%.4f" % (1 / (1 + math.exp(Fspa - Feng))))


if __name__ == "__main__":
    main()
