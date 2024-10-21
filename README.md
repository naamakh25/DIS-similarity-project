Facial Movements and Dynamic Identity Signature Project


This project analyzes facial movements to generate Dynamic Identity Signatures (DIS) using Normalizing Flow models. The objective is to explore patterns and similarities between individuals by analyzing their facial landmarks.

Files in this Repository
1. HeadPose_NF.py
Processes video data to extract head pose angles using Mediapipe. The extracted angles are used to define each participant’s Dynamic Identity Signature (DIS). A Normalizing Flow model is then trained on these angles to capture the dynamics of each participant's facial movements.


2. smile detection.py
Detects smiles in videos by extracting facial blendshapes using Mediapipe. Smile detection is based on predefined thresholds for specific blendshape scores, and the results are saved in two Excel files—one for smile detection per frame and one for smile detection per second.

3. NF_train.py
Trains Normalizing Flow models on facial landmark data extracted from videos. It includes functionality for loading landmarks, training models, saving trained models, and plotting loss values over epochs.

4. NF_with_probabilities.py
Computes log probabilities and norms of facial landmarks for different participants using a trained Normalizing Flow model. Results are saved to Excel, and the norms are visualized in a distribution graph.

5. NF_prepare_general_DIS.py
Processes video data to extract facial landmarks using Mediapipe and saves the landmark data to Excel. Additionally, it uses PCA (Principal Component Analysis) to visualize the landmark distribution in two dimensions. The script processes multiple videos and outputs the results as both Excel files and PCA visualizations. Key features include:

Extracting x and y coordinates for 478 facial landmarks per frame.
Storing the extracted landmarks, timestamp, and frame number in Excel.
Visualizing landmark distribution using PCA and saving the resulting scatter plot.
How to Use
Install dependencies:

Copy code
pip install -r requirements.txt
Processing Videos:

To extract landmarks and generate the DIS, use HeadPose_NF.py or NF_prepare_general_DIS.py.
For smile detection, use smile detection.py.
Training Models:

Use NF_train.py to train Normalizing Flow models on the extracted landmarks.
Comparing Participants:

Use NF_prepare_general_DIS.py to visualize landmark distributions using PCA.
Acknowledgements
This project uses Mediapipe for facial landmark detection and Normalizing Flow models for analyzing and generating Dynamic Identity Signatures (DIS).
