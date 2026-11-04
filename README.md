Overview:Overview
Build an algorithm that recognizes hand gestures from muscle signals—regardless of who is wearing the device.

This competition uses electromyography (EMG) data recorded with Aura VR, a wearable biosignal device developed by PiEEG. The dataset was collected and labeled using Aura Dataset Lab.

Goal

Your task is to train a machine learning model that predicts hand gestures from EMG signals.

Muscle activity differs between individuals. The challenge is to learn patterns that reliably identify gestures across people, rather than patterns specific to individual participants in the training data.

The ultimate goal is a gesture recognition algorithm that can power an Aura VR wearable interface: a person puts on the device, performs a gesture, and the algorithm recognizes it.

Start

7 days ago
Close

a month to go
Description
Recognizing Gestures Across People

Hand gestures produce patterns of electrical activity in the muscles. Electromyography (EMG) captures these signals, allowing machine learning algorithms to identify gestures from wearable sensor recordings.

However, muscle signals vary between people. An algorithm that performs well on one person may struggle with another. This competition challenges participants to develop a gesture recognition algorithm that generalizes to people whose recordings were not used during training.

About the Data

The dataset was collected using Aura VR, a wearable biosignal device developed by PiEEG, with recording and labeling managed through Aura Dataset Lab.

The available data includes:

Raw EMG signals, capturing electrical activity associated with muscle contractions. Inertial measurement unit (IMU) data, providing information about movement through accelerometer and gyroscope measurements. Gesture labels, identifying the target actions in the training recordings.

These complementary signals offer different perspectives on the same action: EMG captures muscle activity, while the IMU captures movement.

Your Approach

You are free to design your own signal processing and modeling pipeline. You may explore filtering, normalization, temporal windowing, feature extraction, traditional machine learning, deep learning, or combinations of these approaches.

You can develop models using EMG, IMU data, or both. Combining the sensor streams may help distinguish gestures and improve robustness across individuals.

The objective is to recognize gestures, rather than identify the person performing them. When developing and validating your approach, consider how well it works on people excluded from model training.

From Recordings to a Wearable Interface

The ultimate goal is an algorithm that can power a practical Aura VR gesture interface: a person puts on the device, performs a hand gesture, and the algorithm recognizes it.

Your solution will help explore how raw muscle and motion signals can support reliable interactions with computers, virtual environments, and assistive technologies.

Evaluation
Submissions are evaluated using macro-averaged F1 score between the predicted gesture labels and the true labels.

F1 combines precision and recall for each class. The macro F1 score is the unweighted average of these scores across all target classes, giving each class equal importance. Scores range from 0 to 1, with higher scores indicating better performance.

To assess recognition across people, the test set will contain participants whose recordings are excluded from the training set.

Submission File
For each ID in the test set, predict a single gesture label. Submit a CSV file containing exactly two columns: ID and gesture.

For example:

ID,gesture
2,fist
5,open
6,pinch
Use the exact label values provided in the training data. Include one prediction for every test ID, with no missing or duplicate IDs and no additional index column.

Follow the structure and ID order in the provided sample_submission.csv.

Prizes
Prizes will be awarded to the top three eligible entries in the final competition standings.

Total Prize Pool: 1,000

1st Place: 1,000 and Certificate

2st Place: Certificate

3st Place: Certificate

Citation
PiEEG. Aura: Calibration-Free EMG Challenge. https://www.kaggle.com/competitions/vr-calibration-free-emg-challenge-pieeg, 2026. Kaggle.

DATA:

Dataset Description
Dataset Description
Dataset version 2 (updated 29 Oct 2026)

Files • train/train_1.xlsx … train_28.xlsx: full recording sessions with labels (unchanged from v1) • test/test_1.xlsx … test_5.xlsx: test actions without labels • sample_submission.csv: example submission (action_id,gesture)

All data files are Excel (.xlsx); the submission is CSV.

Test set format (v2) • Each action_id is one gesture action: a 1.4 s window, rows in time order • t restarts at 0 for each action • Every signal channel is mean-centred per action • Columns: action_id, t, emg_new, imu_new, ch1–ch8, ax, ay, az, gx, gy, gz • roll, pitch, yaw and all camera / hand-tracking columns are not included in test

Train set • Full sessions with all columns, including label and camera / hand-tracking columns • Only rows with label_source = guided correspond to the test actions; other rows are transitions • Tip: to match the test format, cut your training data into 1.4 s windows and mean-centre each channel per window

Task Predict one gesture per action_id: rest, fist, open, point, pinch, thumb-up. Scoring: macro F1.

Final evaluation Top models will be verified live on the Aura device with new users after the competition ends.

Intro
This dataset contains raw electromyography (EMG) signals and inertial measurement unit (IMU) data recorded using Aura VR, a wearable device developed by PiEEG. Recordings and gesture annotations were collected using Aura Dataset Lab.

The task is to predict hand gestures from these sensor recordings. The competition focuses on developing algorithms that generalize across people, including individuals whose recordings were not used during training.

Data Provided
Training data: Sensor recordings and corresponding gesture labels for developing your models.
Test data: Sensor recordings for which you must predict the gesture labels.
Sample submission: A template showing the required prediction format and test identifiers.
Use the training data to develop your approach, the test data to generate predictions, and the sample submission to format your final submission.

Sensor Signals
EMG
EMG measures electrical activity produced by muscles. The raw recordings contain time-varying signals that can be used to learn patterns associated with different hand gestures.

Participants may apply their own filtering, normalization, feature extraction, and other signal processing techniques, or develop deep learning models that learn directly from the signals.

IMU
The IMU provides accelerometer and gyroscope measurements:

Accelerometer: Measures acceleration along three axes, including the contribution of gravity.
Gyroscope: Measures angular velocity around three axes.
These measurements provide motion information that can complement the EMG signals. Participants may explore either sensor type independently or combine them.

Prediction Target
Predict the gesture associated with each test example, using the exact target labels provided in the training data.

Aura Dataset Lab supports the following built-in labels:

Label	Description
fist	Close the hand into a fist.
open	Open the hand and spread the fingers.
pinch	Touch the thumb to the index finger.
point	Extend the index finger while curling the other fingers.
thumb-up	Raise the thumb with the fingers loosely curled.
rest	Relax the hand without deliberately contracting the muscles.
The labels included in the released competition files define the final set of prediction classes.

Working with the Data
The recordings are time-series data, so the order of samples matters. Preserve recording boundaries when constructing temporal windows, and avoid combining samples from different people or sessions into a single window.

To assess generalization across people, use participant-separated validation whenever participant identifiers are available: recordings from the same person should remain entirely within either the training or validation partition.

Gesture annotations and any fields derived from those annotations are targets or supporting information, rather than sensor inputs. Build your prediction pipeline using the inputs available in the test set.

Acronyms
EMG: Electromyography.
IMU: Inertial measurement unit.
ML: Machine learning.
DL: Deep learning.
CSV: Comma-separated values.
Files
34 files

Size
212.81 MB

Type
xlsx, csv

License
MIT

sample_submission.csv(3.85 kB)

2 of 2 columns


action_id

gesture
348

unique values
1

unique value
a0000
rest
a0001
rest
a0002
rest
a0003
rest
a0004
rest
a0005
rest
a0006
rest
a0007
rest
a0008
rest
a0009
rest
a0010
rest
a0011
rest
a0012
rest
a0013
rest
a0014
rest
a0015
rest
a0016
rest
a0017
rest
a0018
rest
a0019
rest
a0020
rest
a0021
rest
a0022
rest
a0023
rest
a0024
rest
a0025
rest
a0026
rest
a0027
rest


Rules:You have accepted the rules for this competition. Good luck!
Competition Rules
Kaggle Competition Foundational Rules
(Non-editable)

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.
The following Kaggle Competition Foundational Rules (“ Foundational Rules ”) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize. b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records. c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible. d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website. b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website. c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules. d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification. b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code. c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set. b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email. b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria. c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph. d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle. e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded. f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants. b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions). c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8. d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed. e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above. f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>. b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law. b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity. c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition. c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard. d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission. e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition. f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition. g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules. h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor. i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.
