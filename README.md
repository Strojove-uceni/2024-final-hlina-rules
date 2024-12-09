[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rMTkWhxv)
*Reminder*
*   *Do not miss [deadline](https://su2.utia.cas.cz/labs.html#projects) for uploading your implementation and documentation of your final project.*
*   *Include working demo in Colab, in which you clearly demonstrate your results. Please refer to the Implementation section on the [labs website](https://su2.utia.cas.cz/labs.html#projects).*
*   *Do not upload huge datasets to GitHub repository.*

* **Detekce anomálií při 3D tisku v reálném čase** *

 **Úvod do problematiky** 

3D tisk je v akademickém i businessovém protředí znám již poměrně dlouhou dobu a je považován za jedno z nejprespektivnějších odvětví průmyslu 4.0. Masová implementace 3D tisku se ovšem i dnes stále střetává s výzvami, které čekají na vyřešení. Jednou z nejvýznamnějších je detekce tiskových chyb a jejich včasná korekce. Zásadními proměnnými při 3D tisku je pozice vstřikovací trysky, rychlost, množství a teplota filamentu (vstřikovaného materiálu). Správná kombinace těchto veličin je klíčovým kritériem dosažení bezchybného tisku a vychýlení byť jen jedné z těchto čtyř hodnot nevyhnutelně vede k tiskovým anomáliím. Cílem našeho projektu je detekovat tiskové anomálie prostřednictvím odhadu čtyřech výše zmíněných veličin.

 **Trénovací data** 

Při trénování našich modelů jsme použili dataset sestavený týmem Cambridgeských vědců (Brion, D., & Pattinson, S. (2022). Data set for “Generalisable 3D printing error detection and correction via multi-head neural networks”. Apollo - University of Cambridge Repository. https://doi.org/10.17863/CAM.84082). Dataset se skládá z 1,272,273 anotovaných fotografií pořízených kamerami umístěnými na trysce 3D tiskárny. Kamera zachycuje pohyb trysky, její pozici vúči výrobku a aplikaci filamentu v reálném čase s frekvení 2,5 Hz. Celkem je takto zaznamenán tisk 192 komponentů různých tvarů, barev a za různých světelných podmínek na osmi různých tiskárnách. Anotace každé fotografie obsahuje klasifikaci pozice trysky, množství, rychlosti a teploty vstřikovaného filamentu. Veličiny jsou klasifikovány jako 'too low', 'good' nebo 'too high'. 
