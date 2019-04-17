#!/bin/bash

stances="Positive Surprised Amused Interested Certain Comfortable"
parts="dev eval"
listener="active"

for stance in $stances
do 
        echo $stance
        for part in $parts
        do
                echo $part
                python vis.py $stance $part $listener
        done
done
