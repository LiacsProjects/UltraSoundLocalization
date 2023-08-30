#include <NewPing.h>

// trigger and echo pins for each sensor
#define SONAR1TRIG 5
#define SONAR1ECHO 6
#define SONAR2TRIG 9
#define SONAR2ECHO 10

#define MAX_DISTANCE 400 // maximum distance for sensors
#define NUM_SONAR 2 // number of sonar sensors

NewPing sonar[NUM_SONAR] = { // array of sonar sensor objects
  NewPing(SONAR1TRIG, SONAR1ECHO, MAX_DISTANCE),
  NewPing(SONAR2TRIG, SONAR2ECHO, MAX_DISTANCE),
};

int distance[NUM_SONAR]; // array stores distances for each
                         // sensor in cm

void setup() {
  Serial.begin(19200);
}

void loop() {
    delay(300);
    updateSonar(); // update the distance array
    Serial.println(distance[0]);
    Serial.println(distance[1]);
}

// takes a new reading from each sensor and updates the
// distance array
void updateSonar() {
  for (int i = 0; i < NUM_SONAR; i++) {
    distance[i] = sonar[i].ping_cm(); // update distance
    // sonar sensors return 0 if no obstacle is detected
    // change distance to max value if no obstacle is detected
    if (distance[i] == 0)
      distance[i] = MAX_DISTANCE;
  }
}