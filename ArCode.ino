#include <Stepper.h>
#include <Servo.h>

// Function Declarations
void dropPill();

// For serial input
String readString;

// Set up Servo Object
Servo mainDoorServo;
int pos = 90;       // track position of servo

// Set up Stepper Motor
const int stepsPerRevolution = 32;      // Number of steps for one revolution internally (no gear reduction)
const int gearReduction = 64;           // Amount of gear reduction 
const int actualStepsPerRev = stepsPerRevolution * gearReduction;   // Number of steps for one output revolution
// 2048 steps per output revolution for our motor

// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);
          
int stepCount = 0;         // number of steps the motor has taken

void setup() {
  // initialize the serial port:
  Serial.begin(9600);
  myStepper.setSpeed(500);
  delay(500);
  mainDoorServo.attach(6);
  mainDoorServo.write(pos);
  delay(1000);
}

void loop() {
  //Use to correct servo position if accidentally moved, + goes right, - goes left
          // myStepper.step((actualStepsPerRev / 32));
          // dropPill();
          // delay(50000000);
           
while (Serial.available()) {
    char c = Serial.read();   //gets one byte from serial buffer
    readString += c;          //makes the String readString
    delay(2);                 //slow looping to allow buffer to fill with next character
}

if (readString.length() > 0) {
    int n = readString.toInt();  //convert readString into a number
    
    switch(n){
      case 0:     // First Box (front), BLUE
        dropPill();
        break;
      case 1:     // Second Box (right side), GREEN
          myStepper.step((actualStepsPerRev / 4));
          delay(500);
          dropPill();
          myStepper.step(-(actualStepsPerRev / 4));
        break;
       case 2:    // Fourth Box (left side), RED
          myStepper.step(-(actualStepsPerRev / 4)- 130);
          delay(500);
          dropPill();
          myStepper.step((actualStepsPerRev / 4) + 130);
        break;
      case 3:     // Third Box (back side), WHITE
          myStepper.step((actualStepsPerRev / 2));
          delay(500);
          dropPill();
          myStepper.step(-(actualStepsPerRev / 2));
        break;
    }

    readString="";
  } 
}


// Moves the servo to drop pill into box beneath
void dropPill() {
  for (pos = 90; pos >= 30; pos -= 1) {
    mainDoorServo.write(pos);              
    delay(15);                       
  }
  
  for (pos = 30; pos <= 90; pos += 1) { 
    mainDoorServo.write(pos);              
    delay(15);                       
  }
  
  delay(500);
}
