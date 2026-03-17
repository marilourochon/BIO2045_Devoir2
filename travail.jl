# ---
# title: Titre du travail
# repository: marilourochon/BIO2045_Devoir2
# auteurs:
#    - nom: Rochon
#      prenom: Marilou
#      matricule: 20237984
#      github: marilourochon
#    - nom: Comte Desjardins
#      prenom: Miya Yuki
#      matricule: 20271611
#      github: miiyayukii
# ---

# # Introduction
#**Situation biologique**
#On veut aménager un site déforesté en dessous de lignes électriques. Le corridor à aménager
#comprend 200 parcelles vides à aménager avec des herbes et des buissons. Nous avons la possibilité 
#d'aménager 50 parcelles avec les espèces de notre choix. À l'équilibre, pour ne pas interférer avec 
#l'infrastructure, un maximum de 20% des parcelles peuvent être végétalisées, c'est à dire 40 
#parcelles. Il faut que 30% des parcelles végétalisées soient des herbes et 70% des buissons,
#donc 12 parcelles au maximum avec des herbes et 28 parcelles au maximum avec des buissons.
#Finalement, il faut que l'espèce la moins abondante de buissons ne représente pas moins de 30% de
#la surface recouverte par les buissons, soit plus de 8 parcelles. 

#**Question**
#Le but de cette simulation est de modéliser cette situation pour identifier une population initiale, 
#c'est à dire le nombre de parcelles à recouvrir de chaque espèce permettant d'atteindre les 
#recouvrements à l'équilibre requis. De plus, il faut identifier une matrice de transition qui permet
#de respecter ces critères dans au moins 80% des simulations. Finalement, ce modèle permettra de 
#comparer les modèles déterministes et les modèles stochastiques.
#Les modèles déterministes sont des modèles dont la solution est déterminée uniquement par les 
#paramètres fixés, et où le hasard n'intervient pas. Dans ce cas, tant que les paramètres ne sont
#pas modifiés, la solution finale sera la même. Les modèles stochastiques sont, quant à eux, des
#modèles basés sur des variables aléatoires (hasard) ou pas des distributions. La solution obtenue 
#variera donc toujours, mais on observa des tendances dans les distributions de fréquences selon la
#situation (Renard et al., 2013). Les modèles déterminstes sont plus facilement impactés par des petits
#changements de valeur des paramètres, ce qui fait que les valeurs ou les solutions obtenues varient
#énormément. Ces modèles sont considérés comme "moins stables" que les modèles stochastiques (Renard et al. 2013).


#**hypothèses et résultats attendus**
#On s'attend à ce que le modèle déterministe donne une distribution de la population entre les différents
#états qui est différente peu importe le nombre de simulations effectuées. On s'attend aussi à ce que 
#la population se stabilise au bout d'un certain nombre de générations pour atteindre l'équilibre.
#On s'attend à ce que le modèle stochastique produise de multiples solutions différentes, qui devraient 
#donner une tendance qui tourne autour de la solution offerte par le modèle déterministe. 
# # Présentation du modèle
#Pour déterminer les changements de la composition de la population à chaque génération, il faudra utiliser
#une matrice de transition et la multiplier par le vacteur des effectifs de chaque état au temps t . 
#Cette matrice est carrée, ses dimensions sont le (n états x n états). Les différentes
#valeurs contenues dans la matrice représentent la probabilité qu'une parcelle change d'un état à 
#un autre au temps t+1. Le vecteur contient les valeurs d'effectifs de n états.

#On suppose que le nombre de parcelles est constant. La somme des effectifs des états devrait toujours être 
#égale au nombre de parcelles. Comme la taille de la population est fixe, la somme des probabilités 
#associées à un état ne peut pas être supérieure à 1. 
# # Implémentation

# ## Packages nécessaires

import Random
Random.seed!(2045)
using CairoMakie
using Distributions

# ##
"""
check_transition_matrix
vérifie que la somme des cellules d'une ligne dans la matrice est égale à 1
si la somme n'est pas égale à 1, renvoie un avertissement
T est une matrice
"""
function check_transition_matrix!(T)
    for ligne in axes(T, 1)
        if sum(T[ligne, :]) != 1
            @warn "La somme de la ligne $(ligne) n'est pas égale à 1 et a été modifiée"
            T[ligne, :] ./= sum(T[ligne, :])
        end
    end
    return T
end

"""
check_function_arguments
vérifie que la matrice de transition soit carrée et que le nombre d'états corresponde à
la matrice de transition. Si ne correspond pas, renvoieun message d'erreur.
transitions correspond à matrice de probabilités de changement d'état, 
states correspond à un vecteur d'état
"""
function check_function_arguments(transitions, states)
    if size(transitions, 1) != size(transitions, 2)
        throw("La matrice de transition n'est pas carrée")
    end

    if size(transitions, 1) != length(states)
        throw("Le nombre d'états ne correspond pas à la matrice de transition")
    end
    return nothing
end

"""
_sim_stochastic
change les états des parcelles de manière stochastique, en fonction de la probabilité qu'elle
change selon la matrice de transition
timeseries correspond à
transitions correspond à une matrice de probabilités de changement d'état
generation correspond à
"""
function _sim_stochastic!(timeseries, transitions, generation)
    for state in axes(timeseries, 1)
        pop_change = rand(Multinomial(timeseries[state, generation], transitions[state, :]))
        timeseries[:, generation+1] .+= pop_change
    end
end

"""
_sim_determ
change les états des parcelles de manière déterministe, donc calcule le nombre de parcelle pour
chaque état selon les probabilités, mais pas de manière aléatoire dans l'espace
timeseries correspond à
transitions correspond à une matrice de probabilités de changement d'état
generation correspond à
"""
function _sim_determ!(timeseries, transitions, generation)
    pop_change = (timeseries[:, generation]' * transitions)'
    timeseries[:, generation+1] .= pop_change
end

"""
simulation
simule les différentes générations
Par défaut, le nombre de générations est de 500 et la simulation n'est pas stochastique
transitions est une matrice de probabilités de transition
states est une matrice des effectifs pour chaque état à la génération actuelle
"""
function simulation(transitions, states; generations=500, stochastic=false)

    check_transition_matrix!(transitions)
    check_function_arguments(transitions, states)

    _data_type = stochastic ? Int64 : Float32
    timeseries = zeros(_data_type, length(states), generations + 1)
    timeseries[:, 1] = states

    _sim_function! = stochastic ? _sim_stochastic! : _sim_determ!

    for generation in Base.OneTo(generations)
        _sim_function!(timeseries, transitions, generation)
    end

    return timeseries
end

# ##
# States : vecteur qui contient les effectifs de chaque état
# Barren, Grass, Shrub1, Shrub2
s = [150, 8, 19, 20]
states = length(s)
patches = sum(s)

# Transitions : matrice qui définit les probabilités des transition de chaque état à un autre état
T = zeros(Float64, states, states)
T[1,:] = [0.98, 0.01, 0.005, 0.005]
T[2,:] = [0.20, 0.75, 0.03, 0.02]
T[3,:] = [0.05, 0.10, 0.80, 0.05]
T[4,:] = [0.05, 0.06, 0.05, 0.84]
T

#définir quel état a quelle position dans la matrice
states_names = ["Barren", "Grasses", "Shrub1", "Shrub2"]
#définir par quelles couleurs seront représentés les différents états dans les graphiques
states_colors = [:grey40, :orange, :teal, :purple]

# ##
# Simulations

f = Figure()
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles")


#on utilise les deux versions de la simulation, soit stochastique et déterministe et on les
# superpose dans un même graphique pour comparer les effets de la stochasticité

# Stochastic simulation
for _ in 1:100
    sto_sim = simulation(T, s; stochastic=true, generations=200)
    for i in eachindex(s)
        lines!(ax, sto_sim[i, :], color=states_colors[i], alpha=0.1)
    end
end

# Deterministic simulation
det_sim = simulation(T, s; stochastic=false, generations=200)
for i in eachindex(s)
    lines!(ax, det_sim[i, :], color=states_colors[i], alpha=1, label=states_names[i], linewidth=4)
end

#création de la figure 
axislegend(ax)
tightlimits!(ax)
current_figure()

#vérification finale nombre de parcelles par état
sim = simulation(T, s; stochastic=true, generations=200)
final = sim[:,end]

barren = final[1]
grass = final[2]
shrub1 = final[3]
shrub2 = final[4]

végétalisé = grass + shrub1 + shrub2
shrubs = shrub1 + shrub2

println("Barren = ", barren)
println("Grass = ", grass)
println("Shrub1 = ", shrub1)
println("Shrub2 = ", shrub2)

println("Végétalisé total = ", végétalisé)
println("Shrubs total = ", shrubs)
println("Min shrub = ", min(shrub1,shrub2))

# # Présentation des résultats

# La figure suivante représente 


# # Discussion

# On peut aussi citer des références dans le document `references.bib`,
# @ermentrout1993cellular -- la bibliographie sera ajoutée automatiquement à la
# fin du document.
