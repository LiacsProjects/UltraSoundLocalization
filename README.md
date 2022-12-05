# UltraSoundLocalization
This repository contains code and documentation about a localization system based on ultrasound.

Authors: Richard van Dijk, Research Software Engineer @ LIACS, and Bas van Aalst bachelor student @ LIACS.

Title: Scalable outdoor Real-Time Localization System RTLS with ultrasound.

Description:
In the Urban computing domain, accurate real-time outdoor localization of moving objects/subjects is challenging. Most applications use Global Positioning Systems (GPS) for urban localization. However, the accuracy of GPS depends on environmental conditions like the weather (atmospheric influences), surrounding structures like buildings and vegetation, or technical system flaws of Global Navigation Satellite Systems  (GNSS). GPS accuracy, therefore, ranges between 1.5 and 10 meters, and cannot be used indoors.
For several outdoor localization applications, there is a need for 10 times higher accuracy ranging between 0.1 and 1 meter. Applications can be found in the social sciences domain, health care, human-machine interaction research, location-based retail marketing services, sports enhancement, and urban navigation support services for the disabled to mention a few.
This study has a relation with the breaking-the-cycle and focus-on-emotions project, a research line of the Centre of BOLD cities. Here we try to localize normal and vulnerable children during their recess periods at school with as goal to improve the social safety of vulnerable children [6].

Research questions:
1.	How can we improve the location positioning system based on ultrasound created by the ludev-nl/2021-18-Indoor-localization team inspired by the state-of-the-art?
2.	How does the system perform compared to an RTLS based on ultra-wideband technology?
3.	How can we extend this system with automatic calibration and setup of the beacons?
4.	How can we scale up this system so it can localize subjects in a larger area than the range of one ultrasound beacon (12-15 meters)?

Plan:
For Research Question 1:
-	Play with the time-series Audacity tool and Frequency Generator app for smartphones to be used as beacons. 
Deliverable: Excel with distance graph.
-	Search for state-of-the-art localization systems based on ultrasound and compare with the sweep-based method developed by the ludev-nl/2021-18-Indoor-localization team. Deliverable: related work chapter for the thesis.
-	Improve the software package that implements the sweep-based ultrasound method developed by the ludev-nl/2021-18-Indoor-localization team, so that it establishes the trajectories of slowly moving objects/subjects. 
Deliverable: software that plots the trajectory based on several recordings of three or four beacon sweeps and methods chapter for the thesis about the sweep-based ultrasound RTLS.


For Research Question 2:
-	Set up the UWB system and record a number of trajectories together with ultrasound sweeps from three or four beacons.
Deliverable: a number of identical trajectory data from UWB and Ultrasound RTLS.
-	Make software to compare the UWB trajectories with the Ultrasound trajectories with a number of distance metrics.
Deliverable: Experiment chapter for the thesis.

For the other two research questions – improved usability and scalability -, we will make a plan at that time.

Profile student:
-	Interests in programming in Python
-	Interests in practical engineering (field) work
-	Interests in sensor systems for real-time localization

References:
[1] Jim´enez, A. R., and Seco F., 
Ultrasonic localization methods for accurate positioning. 
Instituto de Automatica Industrial, Madrid (2005). 
https://www.researchgate.net/profile/Antonio-Jimenez-11/publication/228657454 Ultrasonic Localization Methods for Accurate Positioning/links/09e415093b2da1eff9000000/

[2] Piontek, H., Seyffer, M. and Kaiser, J.,
Improving the accuracy of ultrasound-based localisation systems. 
Pers Ubiquit Comput 11, 439–449, (2007). 
https://doi.org/10.1007/s00779-006-0096-1

[3] Potort`ı, F.; Park, S.; Jim´enez Ruiz, A.R.; Barsocchi, P.; Girolami, M.; Crivello, A.; Lee, S.Y.; Lim, J.H.; Torres-Sospedra, J.; Seco, F.; Montoliu, R.; Mendoza-Silva, G.M.; P´erez Rubio, M.D.C.; Losada-Guti´errez, C.; Espinosa, F.; Macias-Guarasa, J., 
Comparing the Performance of Indoor Localization Systems through the EvAAL Framework. 
Sensors 2017, 17, 2327. 
https://doi.org/10.3390/s17102327

[4] Mirshahi S., Mas O., 
A Novel Distance Measurement Approach Using Shape Matching in Narrow-Band Ultrasonic System. 
IFACPapersOnLine, Volume 48, Issue 3, 2015, Pages 400-405, ISSN 2405-8963, 
https://doi.org/10.1016/j.ifacol.2015.06.114

[5] Li J., Han G., Zhu C., Sun G., 
An Indoor Ultrasonic Positioning System Based on TOA for Internet of Things. 
Mobile Information Systems, vol. 2016, Article ID 4502867, 10 pages, 2016.
https://doi.org/10.1155/2016/4502867

[6]
Breaking the Cycle | Centre for BOLD Cities (centre-for-bold-cities.nl)


