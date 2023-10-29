import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.metrics import confusion_matrix

def pgd_attack(model, x, y, epsilon=10, alpha=1, num_iter=10, targeted=False, num_random_init=0, batch_size=280):
                perturbed_x = tf.identity(x)  # create a copy of the input

                for _ in range(num_iter):
                    with tf.GradientTape() as tape:
                        tape.watch(perturbed_x) # keep track of purturbed_x
                        loss = model(perturbed_x) #calculate loss

                    gradients = tape.gradient(loss, perturbed_x) # calculate gradient of loss relevent to pur_x

                    if targeted:
                        gradients = -gradients

                    perturbed_x = tf.clip_by_value(perturbed_x + alpha * tf.sign(gradients), x - epsilon, x + epsilon) #update purtubate x and clip to stay in a specific range
                    perturbed_x = tf.clip_by_value(perturbed_x, 0, 0.5)  # ensure pixel values are in [0, 1] range

                perturbed_x = tf.stop_gradient(perturbed_x) #stop gradientflow
                return perturbed_x, y

def detect_binary_features(data, threshold=0.05):
    num_samples, num_features = data.shape
    binary_mask = np.zeros(num_features, dtype=bool)

    for feature_idx in range(num_features):
        unique_values = np.unique(data[:, feature_idx])
        unique_ratio = len(unique_values) / num_samples

        if unique_ratio <= threshold:
            binary_mask[feature_idx] = True

    return binary_mask

# Define compute_norm and sel_direction functions
def compute_norm(x, x_adv):
    return np.linalg.norm(x_adv - x)

def sel_direction(x, x_adv, x_adv_p):
    norm1 = compute_norm(x, x_adv)
    norm2 = compute_norm(x, x_adv_p)
    if norm2 > norm1:
        direction = -1
    elif norm2 < norm1:
        direction = 1
    else:
        direction = 0
    return direction

def boundary_attack_tabular(model, x, y, max_iterations=50, step_size=0.005, epsilon=0.01):
    binary_mask = detect_binary_features(x)

    # Initialize the adversarial example with the original input
    x_adv = np.copy(x)
    for _ in range(max_iterations):
        # Generate random perturbations for binary features
        binary_perturbation = np.random.normal(0, 0.1, size=x.shape)
        binary_perturbation *= binary_mask  # Apply the binary mask to select binary features

        # Generate random perturbations for other features
        continuous_perturbation = np.random.normal(0, 0.1, size=x.shape)
        continuous_perturbation *= (1 - binary_mask)  # Apply the inverse of the binary mask

        # Combine binary and other perturbations
        total_perturbation = binary_perturbation + continuous_perturbation

        # Calculate Euclidean distance between x_adv and original input x
        distance = compute_norm(x, x_adv)

        # Normalize perturbation to have unit length
        eta_normalized = total_perturbation / np.linalg.norm(total_perturbation)

        # Project the perturbation onto a sphere around the original input
        eta_sphere = distance * eta_normalized

        # Determine the direction of perturbation using sel_direction
        direction = sel_direction(x, x_adv, x_adv + epsilon * (x - x_adv) + eta_sphere)

        # Make a small movement towards the original input
        x_adv = x_adv  + direction * ( epsilon * eta_sphere ) # Add the projected perturbation to the update rule

        # Clip feature values to appropriate ranges
        x_adv = np.clip(x_adv, 0, 1)  # Binary features
        x_adv = np.clip(x_adv, x.min(axis=0), x.max(axis=0))  # Continuous/integer features

        # Check if the adversarial example has caused a misclassification
        adv_preds = model.predict(x_adv)

        if np.argmax(adv_preds) != np.argmax(y):
            # The adversarial example successfully caused a misclassification
            return x_adv

    # Return the modified adversarial example (if no successful attack)
    return x_adv


def carlini_attack(model, x, y, targeted=False, epsilon=0.9, max_iterations=20, learning_rate=0.01):
                perturbed_x = tf.identity(x)  # create a copy of the input

                for _ in range(max_iterations):
                    with tf.GradientTape() as tape:
                        tape.watch(perturbed_x)
                        prediction = model(perturbed_x)

                        if targeted:
                            target_labels = 1 - y
                            loss = tf.reduce_mean(tf.math.log(1e-30 + (1 - prediction) * (1 - target_labels)))
                        else:
                            loss = tf.reduce_mean(tf.math.log(1e-30 + prediction * y))
                            gradients = tape.gradient(loss, perturbed_x)

                    perturbed_x = perturbed_x + learning_rate * gradients
                    perturbed_x = tf.clip_by_value(perturbed_x, x - epsilon, x + epsilon)
                    perturbed_x = tf.clip_by_value(perturbed_x, 0, 1)  # ensure pixel values are in [0, 1] range

                perturbed_x = tf.stop_gradient(perturbed_x)
                return perturbed_x, y

def main():
    left_co, cent_co,last_co = st.columns([2, 3, 2])
    with cent_co:
        st.image('ts.jpg')

    left_c, cent_c,last_c = st.columns([6, 12, 2])
    with cent_c:
        st.title("Model Evaluator")
   

    attack_option = st.sidebar.radio("Select Attack", ["PGD Attack", "Tabular Boundary Attack","Carlini Attack"])

    if attack_option == "PGD Attack":
        st.title("PGD Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])
        
        if dataset_file:
            data = pd.read_csv(dataset_file)

            y = 'Churn'
            x = [x for x in list(data.columns) if x != y]
            X_train, X_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())

        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])
        
        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply PGD Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

                # Apply the PGD attack
            X_test_pgd, y_test_pgd = pgd_attack(loaded_model, X_test_scaled, y_test)

            # Evaluate the perturbed model
            perturbed_model_accuracy = loaded_model.evaluate(X_test_pgd, y_test_pgd)[1]
            #st.write(f"Perturbed Model Accuracy: {perturbed_model_accuracy}")

            # Display accuracy using a gauge visualization
        
            st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
            st.progress(original_model_accuracy)

            st.write(f"Perturbed Model Accuracy: {round(perturbed_model_accuracy, 2)}")
            st.progress(perturbed_model_accuracy)

            st.title("Suggested Defenses")

            with st.container():
                st.subheader("Adversarial Training")
                st.write("Adversarial training is a machine learning technique that enhances model robustness. It involves exposing a model to adversarial examples, which are subtly modified inputs designed to mislead the model's predictions. By iteratively refining the model's response to such examples, adversarial training fortifies it against potential real-world attacks, making it more reliable and secure.")

            with st.container():
                st.subheader("Stochastic Distillation")
                st.write("It involves training a model on a mixture of clean and adversarial examples, using a stochastic process to generate perturbations. This approach aims to reduce the model's sensitivity to small input changes, ultimately bolstering its resilience against adversarial attacks like PGD.")

            with st.container():
                st.subheader("Feature Squeezing with Adoptive Learning Rate")
                st.write(" This leverages data preprocessing to detect and mitigate adversarial perturbations by squeezing the input space. Additionally, it dynamically adjusts the learning rate during training, which helps the model better adapt to adversarial examples, making it more robust against PGD attacks.")

    if attack_option == "Tabular Boundary Attack":
        st.title("Tabular Boundary Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])
        
        if dataset_file:
            data = pd.read_csv(dataset_file)

            y = 'Churn'
            x = [x for x in list(data.columns) if x != y]
            X_train, X_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())

        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])
        
        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply Boundary Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

                # Apply the PGD attack
            X_test_pgd = boundary_attack_tabular(loaded_model, X_test_scaled, y_test)

            # Evaluate the perturbed model
            perturbed_model_accuracy = loaded_model.evaluate(X_test_pgd, y_test)[1]
            #st.write(f"Perturbed Model Accuracy: {perturbed_model_accuracy}")

            # Display accuracy using a gauge visualization
        
            st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
            st.progress(original_model_accuracy)

            st.write(f"Perturbed Model Accuracy: {round(perturbed_model_accuracy, 2)}")
            st.progress(perturbed_model_accuracy)

            st.title("Suggested Defenses")

            with st.container():
                st.subheader("Adversarial Training")
                st.write("Adversarial training is a machine learning technique that enhances model robustness. It involves exposing a model to adversarial examples, which are subtly modified inputs designed to mislead the model's predictions. By iteratively refining the model's response to such examples, adversarial training fortifies it against potential real-world attacks, making it more reliable and secure.")

            with st.container():
                st.subheader("Ensemble Models with Squeezed Features")
                st.write("Feature squeezing strengthens models against attacks, utilizing rescaling and bit reduction. While testing, three models were introduced: a multilayer perceptron, a wide and deep model, and a CNN churn prediction model. Ensemble models combine predictive power, fortifying the defense against adversarial threats.")


    elif attack_option == "Carlini Attack":
        st.title("Carlini Attack")
        dataset_file = st.file_uploader("Upload Dataset", type=["csv"])
        
        if dataset_file:
            data = pd.read_csv(dataset_file)

            y = 'Churn'
            x = [x for x in list(data.columns) if x != y]
            X_train, X_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_train_combined, y_train_combined = under_sampler.fit_resample(X_train_smote, y_train_smote)

            st.write(y_train_combined.value_counts())

        model_file = st.file_uploader("Upload .h5 Model", type=["h5"])
        
        if model_file:
            temp_model_location = "temp_model.h5"
            with open(temp_model_location, 'wb') as out:
                out.write(model_file.read())

            loaded_model = tf.keras.models.load_model(temp_model_location)
            st.session_state.loaded_model = loaded_model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test

            original_model_accuracy = loaded_model.evaluate(X_test_scaled, y_test)[1]
            st.session_state.original_model_accuracy = original_model_accuracy

            st.write(f"Original Model Accuracy: {original_model_accuracy}")

        if st.button("Apply Carlini Attack"):
            if hasattr(st.session_state, 'loaded_model'):
                loaded_model = st.session_state.loaded_model
                X_test_scaled = st.session_state.X_test_scaled
                y_test = st.session_state.y_test

                X_test_carlini, _ = carlini_attack(loaded_model, X_test_scaled, y_test)
                perturbed_accuracy = loaded_model.evaluate(X_test_carlini, y_test)[1]

            
                st.write(f"Original Model Accuracy:{round(original_model_accuracy, 2)}")
                st.progress(original_model_accuracy)

                st.write(f"Perturbed Model Accuracy: {round(perturbed_accuracy, 2)}")
                st.progress(perturbed_accuracy)

                st.title("Suggested Defenses")

            with st.container():
                st.subheader("Adversarial Training")
                st.write("Adversarial training is a machine learning technique that enhances model robustness. It involves exposing a model to adversarial examples, which are subtly modified inputs designed to mislead the model's predictions. By iteratively refining the model's response to such examples, adversarial training fortifies it against potential real-world attacks, making it more reliable and secure.")

            with st.container():
                st.subheader("Defensive Distillation")
                st.write("This approach enhances adversarial example crafting by leveraging insights from a teacher network. The teacher model initially trains on the original dataset, producing softened probabilities for more confident outputs. The student model learns from the teacher's expertise, aiming to fortify resilience against adversarial perturbations with nuanced insights.")

if __name__ == "__main__":
    main()
