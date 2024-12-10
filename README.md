[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rMTkWhxv)
*Reminder*
*   *Do not miss [deadline](https://su2.utia.cas.cz/labs.html#projects) for uploading your implementation and documentation of your final project.*
*   *Include working demo in Colab, in which you clearly demonstrate your results. Please refer to the Implementation section on the [labs website](https://su2.utia.cas.cz/labs.html#projects).*
*   *Do not upload huge datasets to GitHub repository.*

* **Detekce anomálií při 3D tisku v reálném čase** *

 **Úvod do problematiky** 

3D tisk je v akademickém i businessovém protředí znám již poměrně dlouhou dobu a je považován za jedno z nejprespektivnějších odvětví průmyslu 4.0. Masová implementace 3D tisku se ovšem i dnes stále střetává s výzvami, které čekají na vyřešení. Jednou z nejvýznamnějších je detekce tiskových chyb a jejich včasná korekce. Zásadními proměnnými při 3D tisku je pozice vstřikovací trysky, rychlost vstřikování, množství a teplota filamentu (vstřikovaného materiálu). Správná kombinace těchto veličin je klíčovým kritériem dosažení bezchybného tisku a vychýlení byť jen jedné z těchto čtyř hodnot nevyhnutelně vede k tiskovým anomáliím. Cílem našeho projektu je detekovat tiskové anomálie prostřednictvím detekce nesprávných hodnot všech čtyř výše zmíněných veličin.

**Metodologie**

Vycházíme z předpokladu, že správnost pozice vstřikovací trysky, rychlosti vstřikování, množství a teploty filamentu lze rozpoznat v reálném čase vizuálně. Informace o všech čtyřech veličinách je tedy přenositelná přes vizuální záznam tisku. Pro detekci nesprávných hodnot navrhujeme použití námi nadesignované konvoluční neuronové sítě, která je k tomuto účelu natrénována na fotografiích zachycujících průběh tisku. Podobně postupují například Brion a Pattinson (Brion, D.A.J., Pattinson, S.W. Generalisable 3D printing error detection and correction via multi-head neural networks. Nat Commun 13, 4654 (2022). https://doi.org/10.1038/s41467-022-31985-y). Naši neuronovou síť jsme natrénovali na stejném datasetu, který pro trénování použili i autoři výše zmíněného článku. Mimo námi navrhnuté neuronové sítě jsme pro detekci nesprávných hodnot veličin natrénovali na stejném datasetu také upravenou verzi ResNetu. Výsledky trénovaní naší sítě zhodnotíme srovnáním s výsledky Briona a Pattinsona a s výsledky pro námi upravený ResNet.


**Trénovací data** 

Při trénování našich modelů jsme použili dataset sestavený Brionem a Pattinsonem (Brion, D., & Pattinson, S. (2022). Data set for “Generalisable 3D printing error detection and correction via multi-head neural networks”. Apollo - University of Cambridge Repository. https://doi.org/10.17863/CAM.84082). Dataset se skládá z 1,272,273 anotovaných fotografií pořízených kamerami umístěnými na trysce 3D tiskárny. Kamera zachycuje pohyb trysky, její pozici vůči výrobku a aplikaci filamentu v reálném čase s frekvení 2,5 Hz. Celkem je takto zaznamenán tisk 192 komponentů různých tvarů, barev a za různých světelných podmínek na osmi různých tiskárnách. Anotace každé fotografie obsahuje klasifikaci pozice trysky, množství, rychlosti a teploty vstřikovaného filamentu. Veličiny jsou klasifikovány jako 'too low', 'good' nebo 'too high'. 

**Stručný popis použitých arhitektur**

* 
