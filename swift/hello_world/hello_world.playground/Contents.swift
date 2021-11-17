import UIKit

var greeting = "Hello, playground"
// print(greeting)
// print("goodbye")

// Person
var name = "Bob" // string
var age = 51  // Integer
var weight = 50  // Double
var isOrganDonor = false  // Boolean

// print(weight)

// variable is mutable
weight = 20
// print(weight)

// constant is not mutable
let eyeColor = "Blue"
// eyeColor = "green" will return an error

// declare type
// in constrast to Type inference
var message: String = "This is string"
var num: Int = 8

// num = "20" type cannot be overrided -> type safe

// cannot have the same var name twice
// var num = 20

var fullPerson = eyeColor + " " + message // string concatenation
var newMessage = "Hi my name is \(message)"  // string interpolation
newMessage.append(". and i like this")
print(newMessage)

/*
 multi-line comment
 
 double holds more data than float
 */

print("hello")

var students = 30
var Class = 7

let spC = (students / Class)
let remainder = students % Class //modulo, remainder
print(spC, remainder)

var power = pow(20, 2)
print(power)

// casting
print(Double(students))


// bool
var amIbest: Bool = true
amIbest = false

if (true == false) && (true == true) {
    print("hello")
} else if (true == true) {
    print("yezz")
} else {
    print("bye")
}

// ...
// ...

// logical operators and constants
let allowedEntry = false // constants

if (!allowedEntry) {
    print("GOODBYE")
}

// operators: && and, || or
