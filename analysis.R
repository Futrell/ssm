setwd("~/code/ssm")
rm(list=ls())
library(tidyverse)
library(broom)
library(minpack.lm)


d = read_csv("output/model_evaluations/aggregated_output.csv") 

d %>%
  mutate(diff=grammatical_scores - ungrammatical_scores,
         dev_loss=-dev_loss) %>%
  select(dataset, model_type, step, dev_loss, grammatical_scores, diff) %>%
  gather(measure, value, -step, -model_type, -dataset) %>%
  filter(measure == "diff", dataset %in% c("Reg.0.0.0", "Reg.0.0.1", "Reg.0.0.2", "SL.2.1.0", "SL.2.1.1", "SL.2.1.3", "SP.2.1.0", "SP.2.1.1", "SP.2.1.2","TSL.2.1.0","TSL.2.1.1",  "TSL.2.1.2")) %>%
  ggplot(aes(x=step, y=value, color=model_type)) +
    theme_bw() +
    geom_line() +
    facet_wrap(~dataset, scale="free_y")
    
d = read_csv("aggregated_results.csv") %>%
  filter(lang %in% c("SL.2.1.0", "SP.2.1.0", "TSL.2.1.0"),
         lr == 0.01, 
         batch_size == 4,
         model != "wfsa", model != "quasi_sp2")

# Cohen's d

# batch size 6 is re-run after EOS and logspace
# batch size 5 is pTSL re-run after EOS, but not logspace, and also forward not backward algo
# batch size 7 is on pre-logspace checkout of ssm, for debugging

d %>%
  filter((model %in% c("pfsa", "sl2", "sl2_plus_pfsa", "sl2_times_pfsa", "ptsl2", "ptsl2_plus_pfsa", "ptsl2_times_pfsa"))) %>%
  filter(lang %in% c("SL.2.1.0", "TSL.2.1.0")) %>%
  ggplot(aes(x=step*batch_size, y=paired_diff, color=model, linetype=factor(batch_size))) +
  geom_line() +
  facet_grid(lang~lr, scale="free_y") +
  theme_classic()

d %>%
  #filter(model %in% c("opfsa", "pfsa", "sl2", "sl2_plus_pfsa")) %>%
  filter((model %in% c("pfsa", "sl2", "sl2_plus_pfsa", "sl2_times_pfsa", "ptsl2", "ptsl2_plus_pfsa", "ptsl2_times_pfsa"))) %>%
  filter(lang %in% c("SL.2.1.0", "TSL.2.1.0")) %>%
  ggplot(aes(x=step*batch_size, y=-good_scores, color=model, linetype=factor(batch_size))) +
  geom_line() +
  facet_grid(lang~lr, scale="free_y") +
  theme_classic()

# how to model the loss curves. They are cross-entropies. We have h_initial and h_final.
# Exponential: L(d) = h_final + K*a^{-d}. At d=0, this gives L(0) = h_final + K*1 = h_initial.
# Powerlaw:    L(d) = h_final + K*(d+1)^{-a}. At d=0, this gives L(0) = h_final + K*1 = h_initial.
# Kaplan et al. (2021) posits a power law for Transformers trained on language.

# how to model the good/bad diff curves. These might not asymptote!
# Say they have a *final slope* which might be 0 (if they are asymptoting).
# fit y'(t) = a + exp(-bt).
# then y(t) = \sum_0^t y'(t).
# It's either logistic or blowing up.

fit_exp_curve = function(x, y) {
  # fit a curve of the form y = final + slope*exp(-decay*x)
  
  # initial guesses
  initial0 = y[1]+ .001*rnorm(1)
  final0 = y[length(y)] + .001*rnorm(1)
  slope0 = initial0 - final0 + .001*rnorm(1)
  decay0 = 0.5 + .001*rnorm(1)
  
  model = nls(y ~ final + slope*exp(-decay*x), 
             data=data.frame(x=x, y=y),
             start=list(final=final0, slope=slope0, decay=decay0),
             algorithm="port",
             lower=c(decay=0, slope=0, final=0))
  
  model %>%
    tidy() %>%
    select(term, estimate) %>%
    spread(term, estimate)
}

fit_powerlaw_curve = function(x, y) { fit_exp_curve(log(x+1), y) }





# extensive complexity is the final parameter
# non-extensive complexity is slope/decay.



d %>%
  ggplot(aes(x=step*batch_size, y=bad_scores, color=model, linetype=factor(batch_size))) +
  geom_line() +
  facet_grid(lang~lr, scale="free_y") +
  theme_classic()

d %>%
  ggplot(aes(x=step*batch_size, y=mean_loss, color=model, linetype=factor(batch_size))) +
  #geom_line() +
  stat_smooth() +
  facet_grid(lang~lr, scale="free_y") +
  theme_classic()




tur = read_csv("output/model_evaluations/turkish/sl2_times_ptsl2_bs4_e1_lr0.01.csv")
tur %>%
  gather(measure, value, spearman, pearson, mean_loss, nonconvexity, condition_number) %>%
  ggplot(aes(x=step*batch_size, y=value)) +
    geom_line() +
    facet_wrap(~measure, scale="free_y") +
    theme_classic()

que = read_csv("output/model_evaluations/quechua/sl2_times_mtsl2_bs4_e1_lr0.01.csv")

tur %>%
  gather(measure, value, spearman, pearson, mean_loss, nonconvexity, condition_number) %>%
  ggplot(aes(x=step*batch_size, y=value)) +
  geom_line() +
  facet_wrap(~measure, scale="free_y") +
  theme_classic()

