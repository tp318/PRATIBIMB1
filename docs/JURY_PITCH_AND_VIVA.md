# 🎯 PRATIBIMB: Jury Pitch, Live Demo Script & Viva Cheat Sheet
**SIH 2026 · Problem Statement 26054 · DRDO / IDEX**  
*AI-Enabled Real-Time Digital Twin for MALE UAV Aero Piston Engines*

---

## ⏱️ 1. The 2-Minute Elevator Pitch (Say This Verbatim)

> *"Respected Evaluators, in military MALE UAV operations, powerplant failure is the leading cause of aircraft loss. Today, UAV engine health monitoring is purely threshold-based and reactive: if Cylinder Head Temperature exceeds 105 °C, an alarm sounds.
>
> But there is a fatal flaw: **a fixed redline threshold cannot tell the difference between a hot desert day at sea level and a dying cooling system at 6,000 meters altitude.** By the time a fixed alarm fires, the cylinder is already scorched.
>
> We built **PRATIBIMB**—a physics-informed, real-time Digital Twin modeled on the Rotax 914 aero piston engine. 
>
> Instead of monitoring raw numbers, PRATIBIMB runs a condition-matched, healthy twin in parallel, subjected to the exact same throttle commands and International Standard Atmosphere (ISA) altitude lapse rate. We evaluate **residuals**—the physical gap between the real engine and the healthy twin.
>
> When a fault begins, our 9-class AI detects it in milliseconds with **98.82% accuracy**, our PINN-LSTM regressor projects the remaining flight hours, and our **TreeSHAP explainability engine** tells the pilot exactly which physical channel caused the alert.
>
> Finally, our system provides an airworthiness **GO/NO-GO dispatch clearance** and generates a Directorate of Flight Safety & Airworthiness (DFSA) **Sortie Debrief PDF** with maintenance sign-off blocks. PRATIBIMB doesn't just monitor an engine—it guarantees mission success."*

---

## 🎮 2. The 5-Step Killer Live Demo Flow

Follow this exact sequence on `http://localhost:8001/` during your demonstration:

1. **Step 1: Start Normal Sortie**
   - Click **"Start simulation"**.
   - Point to the dials: *"Notice the engine running at 100 Hz physics, streaming live over WebSockets. See the solid blue line (real engine) overlapping the dashed gray line (healthy twin). Residuals are zero because the engine is healthy."*

2. **Step 2: Demonstrate Atmospheric & Weather Resilience**
   - Drag the **Altitude Slider up to 7,000 meters**.
   - Point to CHT dropping: *"Notice CHT dropped from 85 °C down to 38 °C due to the ISA atmospheric lapse rate (-6.5 °C/1,000 m). But notice the Anomaly Score did not spike! Why? Because the twin compensated for altitude, avoiding the false alarms of legacy systems."*

3. **Step 3: Inject an In-Flight Fault**
   - Click **"Inject fault"** $\rightarrow$ select **COOLING SYSTEM FAILURE** (Severity 0.75).
   - In 3 seconds, watch the CHT diverge from the dashed twin line.
   - Point to the status strip: *"Look at the red alert. The anomaly detector tripped, and the 9-class model diagnosed OVERHEATING with 86.6% confidence."*

4. **Step 4: Show Explainable AI (TreeSHAP)**
   - Scroll to the **TreeSHAP Feature Drivers** chart.
   - Say: *"In military aviation, black-box AI is unacceptable. Look here: TreeSHAP explicitly proves that Cylinder Head Temperature residual contributed +45.8% and Oil Temperature slope contributed +24.9% to this diagnosis. The pilot knows the exact physical root cause."*

5. **Step 5: Generate the DFSA-26054 PDF Sortie Debrief**
   - Click the **"Mission report"** tab $\rightarrow$ click **"🖨️ Generate Flight Debrief (PDF)"**.
   - Show the generated document: *"Here is our official Directorate of Flight Safety & Airworthiness sortie record. It features vector SVG health curves, efficiency deficits, and military sign-off lines for the Flight Line Engineer and CTO."*

---

## 🥊 3. Top 10 Hostile Viva Questions & Bulletproof Answers

#### Q1: "Did you validate this on actual test-bench data from an engine cell?"
> **Answer:**  
> *"No military contractor publishes proprietary test-bench data for tactical UAV engines like the Rotax 914. Therefore, following standard aerospace certification methodology (such as NASA's CMAPSS turbofan standards), we formulated the first-principles thermodynamic and rotational ODEs from the official Rotax 914 Flight Manual, calibrated the parameters, and generated a 1.8-million-row stratified dataset across 60 independent engines with zero data leakage."*

#### Q2: "Why not use simple rule-based thresholds?"
> **Answer:**  
> *"Thresholds are blind to operating context. In an ambient temperature swing from sea level to 7,500 m, engine temperatures drop naturally by over 45 °C. A threshold cannot decouple weather from wear. Our Digital Twin residual approach isolates the true mechanical signature, elevating ROC-AUC from 0.70 to 1.00."*

#### Q3: "Can this AI model run onboard a UAV with limited compute?"
> **Answer:**  
> *"Yes. Our models are lightweight 1D-CNN, BiLSTM, and XGBoost structures designed for edge deployment. Our onboard edge module (`EDGE COMPUTE/`) runs in under 1.0 millisecond on a standard ARM Cortex or NVIDIA Jetson Orin Nano, well within the 20 ms / 50 Hz UAV flight control cycle."*

#### Q4: "What if a sensor breaks or drifts?"
> **Answer:**  
> *"We specifically trained **Fault Class 5: SENSOR_DRIFT_FAILURE**. When an individual sensor drifts, only its single residual channel shifts while physically coupled channels (like oil temperature vs. CHT) remain normal. The model recognizes this uncoupled signature and alerts the ground crew to replace the transducer rather than grounding the powerplant."*

#### Q5: "How do you calculate RUL (Remaining Useful Life)?"
> **Answer:**  
> *"We use a Physics-Informed Neural Network (PINN) combined with a causal LSTM. It takes the sliding window of residual vectors, models the Arrhenius wear degradation trajectory, and outputs bounded flight hours with confidence intervals, ensuring monotonic decay toward the replacement limit."*

#### Q6: "Why doesn't your model dynamically retrain its weights mid-flight?"
> **Answer:**  
> *"In military avionics, dynamic mid-flight retraining is strictly prohibited under DO-178C airworthiness guidelines because unconstrained online learning can cause catastrophic model drift. Our models use validated frozen weights during flight, while post-flight sortie debriefs are archived in the database for supervised depot retraining."*

#### Q7: "What is your dispatch criterion for a GO / NO-GO decision?"
> **Answer:**  
> *"The mission engine checks: $\text{Predicted RUL} \ge \text{Sortie Duration} + \text{20-min Loiter Reserve}$. If the weakest subsystem health drops below 0.35 or active combustion instability is diagnosed, the clearance is immediately stamped NO-GO."*

#### Q8: "How does the system handle high-frequency phenomena like misfires?"
> **Answer:**  
> *"We model discrete 4-stroke cycle torque pulses across the 720° crank angle. A misfire causes an instantaneous torque drop on that specific cylinder and elevated exhaust vibration kurtosis, which our temporal CNN-LSTM classifier flags within 3 sliding windows."*

#### Q9: "What database architecture are you using for blackbox flight recording?"
> **Answer:**  
> *"We use a dual-tier persistence system: high-rate 50 Hz fleet blackbox trajectories stream through memory-mapped arrays for real-time scrubbed replay, while sortie manifests, airworthiness decisions, and maintenance debriefs are archived in transactional SQLite / PostgreSQL tables."*

#### Q10: "What makes your project uniquely suitable for DRDO / IDEX?"
> **Answer:**  
> *"Three reasons: First, it addresses the exact Rotax 914 powerplant used in Indian MALE UAVs. Second, it replaces black-box AI with certified TreeSHAP explainability. Third, it generates standardized DFSA-26054 maintenance handoff records that integrate seamlessly into existing military depot maintenance workflows."*
