

# Libraries --------------------------------------------------------------

library(tidyverse)
library(gt)


# Parameters -------------------------------------------------------------

output_folder = "outputs/Pensamiento-TAF" 

# Data preparation -------------------------------------------------------

DF = read_csv(paste0(output_folder, "/output.csv")) |> 
  mutate(DNI = case_when(
    DNI == "29610753" ~ "29619753",
    TRUE ~ DNI
    ))

write_csv(DF, paste0(output_folder, "/output_corregido.csv"))

# Global grades ----------------------------------------------------------

DF |> ggplot(aes(final_grade)) +
  geom_histogram(bins = 10) +
  theme_minimal()

# Item analysis --------------------------------------------------

NO10 = DF |> filter(final_grade != 10) |> pull(DNI)
DF |> 
  # filter(DNI %in% NO10) |> 
  select(ends_with("c")) |> 
  pivot_longer(everything(), names_to = "item", values_to = "accuracy") |> 
  summarise(accuracy = mean(accuracy), .by = "item")


# Grades pdf -------------------------------------------------------------

DF |> 
  select(DNI, number_of_correct_responses, number_of_errors, number_of_no_responses, final_grade) |> 
  arrange(DNI) |> 
  gt::gt() |> 
  gt::fmt_number(columns = final_grade, decimals = 1) |> 
  gt::cols_label(
    DNI = md("**DNI**"),
    number_of_correct_responses = md("**Aciertos**"),
    number_of_errors = md("**Errores**"),
    number_of_no_responses ~ md("**No respuesta**"),
    final_grade = md("**NOTA**"),
  ) |> 
  gt::gtsave(paste0(output_folder, "/TAF  nota.pdf"))
