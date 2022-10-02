# CSE_5243 HW2 

In the folder, there are:

```
HW2-p5.py and utils.py - source code 

report.pdf - a short report describing the approach/procedure I took to construct a feature vector, 
and where applicable the rationale for doing it 

README.md - A detailed README file that contains all the information about the folder
```

The way of running my program:
```
python HW2-p5.py
```

The way of interpreting the output of my code:

In the program, I sampled 5 input sentences and show their non-zero features after preprocessing.
Specifically, I would first output the input sentence and then output lines with the word index, the word, and the frequency. 
An example is shown below:

```
Good service, very clean, and inexpensive, to boot!
        7       to      1.0
        20      good    1.0
        72      and     1.0
        78      very    1.0
        523     service 1.0
        920     inexpensive     1.0
        1667    boot    1.0
        4503    clean   1.0
On the negative, it's insipid enough to cause regret for another 2 hours of life wasted in front of the screen.  
        5       for     1.0
        7       to      1.0
        10      in      1.0
        12      the     2.0
        55      of      2.0
        67      wasted  1.0
        135     on      1.0
        177     it's    1.0
        231     life    1.0
        271     enough  1.0
        343     front   1.0
        389     2       1.0
        567     hours   1.0
        575     screen  1.0
        1045    another 1.0
        1148    regret  1.0
        2052    negative        1.0
        2053    insipid 1.0
        2054    cause   1.0
Damian is so talented and versatile in so many ways of writing and portraying different Characters on screen.  
        0       so      2.0
        2       is      1.0
        10      in      1.0
        55      of      1.0
        72      and     2.0
        135     on      1.0
        332     different       1.0
        575     screen  1.0
        869     many    1.0
        1906    characters      1.0
        2019    talented        1.0
        2123    writing 1.0
        2741    damian  1.0
        2742    versatile       1.0
        2743    ways    1.0
        2744    portraying      1.0
Everything is appalling.  
        2       is      1.0
        295     everything      1.0
        2352    appalling       1.0
The only thing I did like was the prime rib and dessert section.
        12      the     2.0
        15      i       1.0
        72      and     1.0
        77      was     1.0
        201     like    1.0
        247     only    1.0
        294     thing   1.0
        524     did     1.0
        1498    prime   1.0
        4332    rib     1.0
        4333    dessert 1.0
        4334    section 1.0
```
