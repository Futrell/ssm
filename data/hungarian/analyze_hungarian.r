setwd("/Users/hutengdai/Documents/projects/alternation-learning/data/hungarian/")
# Load required packages
library(tidyverse)

# Define the function to label vowels
label_vowel <- function(stem) {
  # Define the vowels for each category
  back_vowels <- c("a", "á", "o", "ó", "u", "ú")
  front_rounded_vowels <- c("ö", "ő", "ü", "ű")
  front_unrounded_vowels <- c("e", "é", "i", "í")
  
  # Identify vowels in the stem and label them accordingly
  back_label <- ifelse(any(str_detect(stem, back_vowels)), "B", "")
  front_rounded_label <- ifelse(any(str_detect(stem, front_rounded_vowels)), "F", "")
  
  # Count the number of front unrounded vowels and create the label
  front_unrounded_count <- sum(str_count(stem, paste(front_unrounded_vowels, collapse = "|")))
  front_unrounded_label <- strrep("N", front_unrounded_count)
  
  label <- paste0(back_label, front_rounded_label, front_unrounded_label)
  
  return(label)
}


# Reading data, creating vowel labels, and adding the match_status column
data <- read.table("wug_test_result.txt", header = TRUE, sep = "\t") %>%
  mutate(
    vowel_label = map_chr(stem, label_vowel),
    match_status = ifelse(predicted_nAk == option_chosen, 1, 0)
  )


# 1. Calculate match status for each subject and stem
data <- data %>%
  mutate(match_status = ifelse(predicted_nAk == option_chosen, 1, 0))

# 2. Summarize match count for each subject
subject_summary <- data %>%
  group_by(subject_code) %>%
  summarize(
    total_stems = n_distinct(stem),
    match_count = sum(match_status)
  ) %>%
  ungroup()

# 3. Determine category for each subject
subject_summary <- subject_summary %>%
  mutate(
    category = case_when(
      match_count == total_stems ~ "All Correct",
      match_count == 0 ~ "All Incorrect",
      TRUE ~ "Mixed"
    )
  )

# Summarize the number of subjects in each category
category_counts <- subject_summary %>%
  count(category)

# Print the results
print(category_counts)

# Calculate the match rate for each vowel label
vowel_label_match_rates <- data %>%
  mutate(vowel_label = map_chr(stem, label_vowel)) %>% 
  group_by(vowel_label) %>%
  summarize(match_rate = mean(match_status)) %>%
  ungroup()



# Assuming your data has columns named 'subject_code' and 'subject_age'
# Group by subject and calculate match rate for each subject
subject_match_rates <- data %>%
  group_by(subject_code, subject_age) %>%
  summarize(match_rate = mean(match_status)) %>%
  ungroup()

# First, create a factor column for age groups. For simplicity, let's assume three age groups: "20-39", "40-59", "60+".
# Adjust these bins according to your dataset.
data <- data %>%
  mutate(
    age_group = cut(subject_age,
                    breaks = c(0, 20, 40, 60, Inf), 
                    labels = c("<20", "20-39", "40-59", "60+"),
                    right = FALSE)
  )

# Then, compute the match rate for each subject
subject_match_rates <- data %>%
  group_by(subject_code, age_group) %>%
  summarize(match_rate = mean(match_status, na.rm = TRUE)) %>%
  ungroup()

# Create a histogram of match rates, faceted by age group
p_histogram <- ggplot(subject_match_rates, aes(x = match_rate)) +
  geom_histogram(binwidth = 0.05, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(
    title = "Distribution of Match Rates Across Subjects by Age Group",
    x = "Match Rate",
    y = "Number of Subjects"
  ) +
  facet_wrap(~ age_group) +
  theme_bw() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Display the plot
print(p_histogram)


average_match_rate <- data %>%
  summarize(average_rate = mean(match_status)) %>%
  pull(average_rate)

print(average_match_rate)


# Calculate the match rate for each subject
subject_match_rates <- data %>%
  group_by(subject_code) %>%
  summarize(match_rate = mean(match_status)) %>%
  ungroup()

# Create a horizontal violin plot of match rates
p_violin <- ggplot(subject_match_rates, aes(y = 1, x = match_rate)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width=0.2, fill="white", orientation="y") + # Flips the boxplot
  labs(
    title = "Distribution of Match Rates Across Subjects",
    x = "Match Rate"
  ) +
  theme_bw() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16),
    axis.title.y=element_blank(),    # Hide the y-axis title
    axis.text.y=element_blank(),     # Hide the y-axis text
    axis.ticks.y=element_blank()     # Hide the y-axis ticks
  )

# Display the plot
print(p_violin)

# Create a histogram of match rates
p_histogram <- ggplot(subject_match_rates, aes(x = match_rate)) +
  geom_histogram(binwidth = 0.05, fill = "skyblue", color = "black", alpha = 0.7) + # binwidth controls the width of each bar
  labs(
    title = "Distribution of Match Rates Across Subjects",
    x = "Match Rate",
    y = "Number of Subjects"
  ) +
  theme_bw() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Display the plot
print(p_histogram)



# Boxplot of match rates by vowel_label
p <- ggplot(vowel_label_match_rates, aes(x = vowel_label, y = match_rate)) + 
  geom_boxplot(outlier.shape = NA) +
  labs(
    y = "Accuracy Rate",
    x = "Vowel Label"
  ) +
  theme_bw() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.background = element_blank(),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Print the plot
print(p)


# Group data by subject and summarize
subject_summary <- data %>%
  group_by(subject_code) %>%
  summarize(
    subject_age = subject_age,
    subject_sex = subject_sex,
    total_responses = n(),
    correct_responses = sum(match_status),
    incorrect_responses = total_responses - correct_responses
  ) %>%
  ungroup() %>%
  mutate(performance_category = case_when(
    correct_responses == total_responses ~ "All Correct",
    incorrect_responses == total_responses ~ "All Incorrect",
    TRUE ~ "Mixed"
  ))



# Calculate the percentage of correct responses for each subject
subject_summary <- subject_summary %>%
  mutate(percentage_correct = (correct_responses / total_responses) * 100)

p_smoothed <- ggplot(subject_summary, aes(x = subject_age, y = percentage_correct, color = subject_sex)) + 
  geom_point(position = position_jitter(width = 0.3, height = 0), alpha = 0.6, size = 2) +
  geom_smooth(method = "loess", aes(group = subject_sex), se = FALSE, size = 1.5) +
  labs(
    title = "Distribution of Age and Sex by Correctness Percentage",
    y = "Percentage Correct",
    x = "Age"
  ) +
  theme_bw() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.background = element_blank(),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Display the plot
print(p_smoothed)

# Count the number of subjects in each performance category
performance_distribution <- subject_summary %>%
  group_by(performance_category) %>%
  summarize(count = n()) %>%
  ungroup()

# Display the result
print(performance_distribution)

# Assuming your data has columns named 'age' and 'sex'
# Group data by subject, age, sex, and summarize
subject_summary <- data %>%
  group_by(subject_code, subject_age, subject_sex) %>%
  summarize(
    total_responses = n(),
    correct_responses = sum(match_status),
    incorrect_responses = total_responses - correct_responses
  ) %>%
  ungroup() %>%
  mutate(performance_category = case_when(
    correct_responses == total_responses ~ "All Correct",
    incorrect_responses == total_responses ~ "All Incorrect",
    TRUE ~ "Mixed"
  ))

# Age distribution for each performance category
age_distribution <- subject_summary %>%
  group_by(performance_category, subject_age) %>%
  summarize(count = n()) %>%
  ungroup()

print(age_distribution)

# Sex distribution for each performance category
sex_distribution <- subject_summary %>%
  group_by(performance_category, subject_sex) %>%
  summarize(count = n()) %>%
  ungroup()

print(sex_distribution)

# Plotting
# Reorder the levels of the 'performance_category' column
subject_summary$performance_category <- factor(subject_summary$performance_category, 
                                               levels = c("All Correct", "Mixed", "All Incorrect"))

# Plotting
p_dot <- ggplot(subject_summary, aes(x = subject_age, y = performance_category, color = subject_sex)) + 
  geom_point(position = position_jitter(width = 0.3, height = 0.2), alpha = 0.6, size = 3) + 
  labs(
    title = "Distribution of Age and Sex by Performance Category",
    y = "Performance Category",
    x = "Age"
  ) +
  theme_bw() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.background = element_blank(),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Display the plot
print(p_dot)




# Group data by subject and vowel_label and summarize
subject_vowel_summary <- data %>%
  group_by(subject_code, vowel_label) %>%
  summarize(
    total_responses = n(),
    correct_responses = sum(match_status),
    incorrect_responses = total_responses - correct_responses
  ) %>%
  ungroup() %>%
  mutate(performance_category = case_when(
    correct_responses == total_responses ~ "All Correct",
    incorrect_responses == total_responses ~ "All Incorrect",
    TRUE ~ "Mixed"
  ))

# Count the number of subjects in each performance category for each vowel label
performance_distribution_vowel <- subject_vowel_summary %>%
  group_by(vowel_label, performance_category) %>%
  summarize(count = n()) %>%
  ungroup()

# Display the result
print(performance_distribution_vowel)

# Optional: Visualize the result
ggplot(performance_distribution_vowel, aes(x = vowel_label, y = count, fill = performance_category)) + 
  geom_bar(stat="identity", position = "dodge") +
  labs(title = "Performance Distribution by Vowel Label") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


mismatched_data <- data %>%
  filter(option_chosen != predicted_nAk & !(nak_advantage >= -1 & nak_advantage <= 1)) %>%
  select(subject_code, subject_age, subject_sex, stem, predicted_sr, predicted_nAk, option_chosen , 
         nak_rating, nek_rating, nak_advantage) %>%
  arrange(subject_age, subject_sex)

mismatched_data <- mismatched_data %>%
  mutate(vowel_label = map_chr(stem, label_vowel))
# Updating the calculation of match rate to exclude the specified stems
data <- data %>% 
  filter(!(option_chosen != predicted_nAk & nak_advantage >= -2 & nak_advantage <= 2))

# Create a binary variable for matches
data$match_status <- ifelse(data$predicted_nAk == data$option_chosen, 1, 0)

# Calculate the match rate for each individual
individual_match_rates <- data %>%
  group_by(subject_code, subject_age, subject_sex) %>%
  summarize(match_rate = mean(match_status)) %>%
  ungroup()


# Save mismatched_data to "mismatch.txt"
write.table(mismatched_data, "mismatch_cases.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 1. Count how many data rows have BNN as the vowel label
BNN_count <- mismatched_data %>%
  filter(vowel_label == "BNN") %>%
  nrow()

cat("Number of data rows with BNN as vowel label:", BNN_count, "\n")

# 2. Calculate what percentage of the mismatched_data dataframe has BNN as the vowel label
total_rows <- nrow(mismatched_data)
BNN_percentage <- (BNN_count / total_rows) * 100

cat("Percentage of data with BNN as vowel label:", BNN_percentage, "%\n")

# 3. Report the overall accuracy excluding the data with BNN as the vowel label
accuracy_without_BNN <- mismatched_data %>%
  filter(vowel_label != "BNN") %>%
  summarize(accuracy = mean(match_status)) %>%
  pull(accuracy)

cat("Overall accuracy without BNN:", accuracy_without_BNN * 100, "%\n")






# Bin subjects into age groups of 20 years
individual_match_rates$age_group <- cut(individual_match_rates$subject_age, 
                                        breaks = seq(0, max(individual_match_rates$subject_age, na.rm = TRUE) + 20, 20),
                                        include.lowest = TRUE, 
                                        labels = c("0-19", "20-39", "40-59", "60-79"))

# Calculate median match rate for annotation
annotation_data <- individual_match_rates %>%
  group_by(age_group, subject_sex) %>%
  summarize(median_rate = median(match_rate)) %>%
  ungroup()

# Create the plot object
p <- ggplot(individual_match_rates, aes(x = age_group, y = match_rate, fill = subject_sex)) +
  geom_boxplot(position = position_dodge(width = 0.75), outlier.shape = NA) +
  geom_text(data = annotation_data, aes(label = sprintf("%.2f", median_rate), y = median_rate), 
            position = position_dodge(width = 0.75), vjust = -1, size = 6, family = "Times New Roman") +
  labs(
    y = "Accuracy Rate",
    x = "Age Group") +
  theme_bw() +  # Use theme_classic as the base
  scale_fill_manual(values = c("gray90", "gray60"), 
                    name = "Sex",
                    breaks = c("male", "female"),
                    labels = c("Male", "Female")) +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.background = element_blank(),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )


print(p)

p_save <- p + 
  theme(
    text = element_text(family = "Times New Roman", size = 14),
    legend.text = element_text(size = 18),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 18)
  )

ggsave(filename = "hungarian-wug.png", plot = p_save, width = 10, height = 7, dpi = 600)


# Calculate the match rate for each individual along with the total and match counts
individual_match_summary <- data %>%
  group_by(subject_code, subject_age, subject_sex) %>%
  summarize(
    total_count = n(),
    match_count = sum(match_status),
    match_rate = mean(match_status)
  ) %>%
  ungroup()

# Bin subjects into age groups of 20 years
individual_match_summary$age_group <- cut(individual_match_summary$subject_age, 
                                          breaks = seq(0, max(individual_match_summary$subject_age, na.rm = TRUE) + 20, 20),
                                          include.lowest = TRUE, 
                                          labels = c("0-19", "20-39", "40-59", "60-79"))

# Generate the summarized table
summary_table <- individual_match_summary %>%
  group_by(age_group, subject_sex) %>%
  summarize(
    total_counts = sum(total_count),
    total_matches = sum(match_count),
    avg_match_rate = mean(match_rate)
  ) %>%
  ungroup()

# Display the summary table
print(summary_table)

# Filter the rows where option_chosen and predicted_nAk do not match, select specific columns, and then sort
mismatched_data <- data %>%
  filter(option_chosen != predicted_nAk) %>%
  select(subject_code, subject_age, subject_sex, stem, predicted_sr, predicted_nAk, option_chosen , 
         nak_rating, nek_rating, nak_advantage) %>%
  arrange(subject_age, subject_sex)

# Save mismatched_data to "mismatch.txt"
write.table(mismatched_data, "mismatch_cases.txt", sep = "\t", row.names = FALSE, quote = FALSE)


# Merge the datasets to include the vowel_label
individual_match_rates <- individual_match_rates %>%
  left_join(select(mismatched_data, subject_code, vowel_label), by = "subject_code")

# Create the plot object with faceting by vowel_label
p <- ggplot(individual_match_rates, aes(x = age_group, y = match_rate, fill = subject_sex)) +
  geom_boxplot(position = position_dodge(width = 0.75), outlier.shape = NA) +
  geom_text(data = annotation_data, aes(label = sprintf("%.2f", median_rate), y = median_rate), 
            position = position_dodge(width = 0.75), vjust = -1, size = 6, family = "Times New Roman") +
  labs(
    y = "Accuracy Rate",
    x = "Age Group") +
  theme_bw() +  # Use theme_classic as the base
  scale_fill_manual(values = c("gray90", "gray60"), 
                    name = "Sex",
                    breaks = c("male", "female"),
                    labels = c("Male", "Female")) +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.background = element_blank(),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  ) +
  facet_wrap(~ vowel_label, scales = "free", ncol = 1) # Facet by vowel_label

# Print the plot
print(p)

# Boxplot of match rates by vowel_label
p <- ggplot(mismatched_data, aes(x = vowel_label, y = match_rate)) +
  geom_boxplot(outlier.shape = NA) +
  labs(
    y = "Accuracy Rate",
    x = "Vowel Label"
  ) +
  theme_bw() +  # Use theme_classic as the base
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.line = element_line(size = 0.1),
    axis.ticks.length = unit(0.15, "inches"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.box.background = element_rect(colour = "black", fill = NA),
    legend.background = element_blank(),
    legend.text = element_text(size = 16),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Print the plot
print(p)


# new plot 

# ANOVA Test
anova_result <- aov(match_rate ~ age_group * subject_sex, data = individual_match_rates)
summary(anova_result)

# Visualization
p2 <- ggplot(individual_match_rates, aes(x = reorder(subject_code, subject_age), y = match_rate, color = subject_sex)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE) +  # Add a smooth trend line without the shaded region
  labs(
    y = "Accuracy Rate",
    x = "Individuals (ordered by age)",
    title = "Accuracy Rate by Age and Gender"
  ) +
  scale_color_manual(values = c("gray60", "gray90"),
                     name = "Sex",
                     breaks = c("male", "female"),
                     labels = c("Male", "Female")) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.text.x = element_blank()  # Hide x-axis labels for clarity
  )

print(p2)




# Visualization
p2 <- ggplot(individual_match_rates, aes(x = reorder(subject_code, subject_age), y = match_rate, color = subject_sex)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE) +  # Add a smooth trend line without the shaded region
  labs(
    y = "Accuracy Rate",
    x = "Individuals (ordered by age)",
    title = "Accuracy Rate by Age and Gender"
  ) +
  scale_color_manual(values = c("gray60", "gray90"),
                     name = "Sex",
                     breaks = c("male", "female"),
                     labels = c("Male", "Female")) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 12),
    axis.text.x = element_blank()  # Hide x-axis labels for clarity
  )

print(p2)
# Visualization with exact age and separate lines for each gender using different shapes
p3 <- ggplot(individual_match_rates, aes(x = subject_age, y = match_rate, color = subject_sex, shape = subject_sex)) +
  geom_jitter(width = 0.3, alpha = 0.6, size = 3) + # Jittered points for clarity
  geom_smooth(aes(group = subject_sex), method = "loess", se = FALSE) +  # Separate smooth lines for each gender
  labs(
    y = "Accuracy Rate",
    x = "Age",
    title = "Accuracy Rate by Exact Age and Gender"
  ) +
  scale_color_manual(values = c("gray60", "gray90"),
                     name = "Sex",
                     breaks = c("male", "female"),
                     labels = c("Male", "Female")) +
  scale_shape_manual(values = c(16, 17), # Use different shapes for genders
                     name = "Sex",
                     breaks = c("male", "female"),
                     labels = c("Male", "Female")) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 12)
  )

print(p3)


p3 <- ggplot(individual_match_rates, aes(x = subject_age, y = match_rate, color = subject_sex, shape = subject_sex)) +
  geom_jitter(width = 0.3, alpha = 0.6, size = 3) + # Jittered points for clarity
  geom_smooth(aes(group = subject_sex), method = "loess", se = FALSE) +  # Separate smooth lines for each gender
  labs(
    y = "Accuracy Rate",
    x = "Age",
    title = "Accuracy Rate by Exact Age and Gender"
  ) +
  scale_color_manual(values = c("gray60", "gray90"),
                     name = "Sex",
                     breaks = c("male", "female"),
                     labels = c("Male", "Female")) +
  scale_shape_manual(values = c(16, 17), # Use different shapes for genders
                     name = "Sex",
                     breaks = c("male", "female"),
                     labels = c("Male", "Female")) +
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 12)
  )

print(p3)


# Generating new predictions from the model
fitted_values <- fitted(bfit_raw_age, newdata = newdata)

# Extract the median and credible intervals
predictions_summary <- data.frame(
  subject_age = newdata$subject_age,
  subject_sex = newdata$subject_sex,
  predicted = rowMeans(fitted_values),
  lwr = apply(fitted_values, 1, function(x) quantile(x, 0.025)),
  upr = apply(fitted_values, 1, function(x) quantile(x, 0.975))
)

# Now, let's try the plot again
ggplot(predictions_summary, aes(x = subject_age, y = predicted, color = subject_sex)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr, fill = subject_sex), alpha = 0.2) +
  geom_line() +
  theme_minimal() +
  labs(
    title = "Predicted Accuracy Rate by Exact Age and Gender",
    x = "Age",
    y = "Predicted Accuracy Rate",
    color = "Gender",
    fill = "Gender"
  )



# Fit the Bayesian linear regression model
# bfit <- brm(match_rate ~ age_group * subject_sex, data = individual_match_rates, family = gaussian(), iter = 4000, chains = 4)
bfit_raw_age <- brm(match_rate ~ subject_age * subject_sex, data = individual_match_rates, family = gaussian(), iter = 4000, chains = 4)

# Summary of the model
# summary(bfit)
summary(bfit_raw_age)

# Plot the effects
# plot(bfit)
plot(bfit_raw_age)

# Using brms to fit a Bayesian GAM with interaction between smooth function and gender
bfit_gam <- brm(
  match_rate ~ s(subject_age, by = subject_sex) + subject_sex, 
  data = individual_match_rates, 
  family = gaussian(), 
  iter = 4000, 
  chains = 4
)

# Summary of the model
summary(bfit_gam)

plot(bfit_gam)



ggplot(individual_match_rates, aes(x = subject_age, y = match_rate, color = subject_sex)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, aes(group = subject_sex)) +  
  labs(title = "Accuracy Rate by Exact Age and Gender", y = "Accuracy Rate", x = "Age") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman", size = 12))

ggplot(individual_match_rates, aes(x = match_rate, fill = subject_sex)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~age_group) +
  labs(title = "Density of Accuracy Rate by Age Group and Gender", y = "Density", x = "Accuracy Rate") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman", size = 12))

lm_fit <- lm(match_rate ~ subject_age * subject_sex, data = individual_match_rates)
summary(lm_fit)

anova_fit <- aov(match_rate ~ age_group * subject_sex, data = individual_match_rates)
summary(anova_fit)
