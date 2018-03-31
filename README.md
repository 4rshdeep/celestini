# Celestini - Phase I (2018)

This github repository contains the solution for the the take away assignment for phase I of Celestini.

## How to check our solution?

1. Begin by opening the file `answers.pdf`. It contains all the answers which didn't involve coding, like the MCQs and our design on solving a socio-economic problem.
2. The answers of question 1 are written in bold just after each question.
3. The second question which was on sparse matrix operations had 3 parts. The answer to the first part is python code which is present inside the folder `programming1`. The answers to second and third part are present inside `answers.pdf` itself. We haven't created a separate readme for this as we could explain it better by writing comments in the code directly.
4. The third question has been answered in the exact same way as question 2.
5. The fourth question had 2 parts, out of which we had to do only one. We did the first one. The code for it is written in a jupyter notebook `cryptosystem_identifier.ipynb` which is present inside the folder `cryptosystem_identifier`. We also wanted to try the second question on Arduino, as neither of us has done robotics before and learning answers new is always fun. But unfortunately, we didn't have the time to do it because of our exams.
6. The answer to the fifth question is written in `answers.pdf`.

### How to run our code?

1. The code for question 2 can be run by typing `python p1.py`. It would ask for the value of N (square matrix dimension). Input both the matrices by inputting each row (space seperated columns) in a seperate line.
```
>>> python p1.py
Enter the value of n: 3
Input for first 3 * 3 matrix
Enter row  0  (space seperated values)
0 1 0
Enter row  1  (space seperated values)
3 0 0..
Enter row  2  (space seperated values)
0 0 2
Input for second 3 * 3 matrix
Enter row  0  (space seperated values)
1 0 0
Enter row  1  (space seperated values)
0 1 0
Enter row  2  (space seperated values)
0 0 1
Sparse multiplication:  [[(1, 1)], [(0, 3)], [(2, 2)]]
Convolution:  [2]

```
2. The code for question 3 can be run by typing `python p2.py`. It would ask for m and n(the number of rows and columns respectively). Then input the matrix by writing each row on a seperate line(space seperated columns). Finally enter the search value.
```
>>> python p2.py
enter m: 3 
enter n: 2
Enter elements of row  0 (space seperated) 
1 2
Enter elements of row  1 (space seperated)
3 4
Enter elements of row  2 (space seperated)
5 6
enter search value: 4
true
```
3. The code for question 4 is present inside the notebook `cryptosystem_identifier.ipynb`. You could view the notebook as rendered by Github (we have also added an html file `cryptosystem_identifier.html` which can be viewed [here](http://www.cse.iitd.ac.in/~cs5160625/cryptosystem_identifier.html) ) or else run the notebook by typing `jupyter notebook` in terminal. Open our notebook from the browser and run each segment. Our code requires the packages:
    * tensorflow
    * numpy
    * sklearn
    * matplotlib
    * seaborn
    * jupyter

## Authors

* **Mayank Singh Chauhan** - [mayanksingh2298](https://github.com/mayanksingh2298)
* **Arshdeep Singh** - [4rshdeep](https://github.com/4rshdeep)


