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
#modèles basés sur des variables aléatoires (hasard) ou des distributions. La solution obtenue 
#variera donc toujours, mais on observa des tendances dans les distributions de fréquences (Renard et al., 2013).
#Les modèles déterminstes sont plus facilement impactés par des petits changements de valeur des paramètres, ce qui 
#fait que les valeurs ou les solutions obtenues varient énormément. Ces modèles sont considérés comme "moins stables" 
#que les modèles stochastiques (Renard et al. 2013).


#**Hypothèses et résultats attendus**
#On s'attend à ce que le modèle déterministe donne une distribution de la population entre les différents
#états qui, pour les mêmes paramètres, est la même peu importe le nombre de simulations effectuées. On s'attend 
#aussi à ce que la population se stabilise au bout d'un certain nombre de générations pour atteindre l'équilibre, et ce, 
#pour le modèle stochastique comme pour le modèle déterministe.
#On s'attend à ce que le modèle stochastique produise de multiples solutions différentes, qui pour les mêmes paramètres initiaux,
#devraient produire une tendance autour de la solution offerte par le modèle déterministe. 

# # **Présentation du modèle**
#Pour déterminer les changements de la composition de la population à chaque génération, il faudra utiliser
#une matrice de transition basée sur le modèle de Markov et la multiplier par un vecteur d'état au temps t, soit 
# un vecteur contenant le nombre de parcelles par état.

#Puisque les différentes valeurs contenues dans la matrice représentent la probabilité qu'une parcelle change d'un état à 
#un autre au temps t+1, elle est toujours carrée. Ses dimensions sont de n états x n états. Le vecteur d'états aura 
#quant à lui une longueur de n états. **Il faudra s'assurer que cette condition est toujours respectée**. 

#On suppose que le nombre de parcelles est toujours constant. La somme des effectifs des états devrait toujours être 
#égale au nombre de parcelles. De plus, comme la taille des parcelles est fixe, la somme des probabilités 
#associées à un état (ligne dans la matrice) ne peut pas être supérieure à 1. **Il faudra s'assurer que cette condition est
#toujours respectée**.

# ## Implémentation

# ## Packages nécessaires

import Random
Random.seed!(2045)
using CairoMakie
using Distributions

# ## **Fonctions utilisées**

#Comme mentionné précédemment, puisque le nombre de parcelles est fixe, il faudra s'assurer que 
#la somme des probabilités associées à un état ne dépasse pas 1. Si la somme était inférieure à 1, le nombre de 
#parcelles diminuerait. Si la somme était supérieure à 1, le nombre de parcelles augmenterait. Pour ce faire, on 
#s'assure de la somme de la ligne de chaque état dans la matrice est exactement égale à 1. Cette fontion vérifiera 
#ces conditions.
"""
check_transition_matrix
vérifie que la somme d'une ligne dans la matrice est égale à 1
si la somme n'est pas égale à 1, renvoie un avertissement
T est une matrice de transition
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

#Comme mentionné précédemment, il faut s'assurer que les dimensions de la matrice corresponde au vecteur d'états,
#et que la matrice de transition soit carrée. Cette fonction vérifiera ces conditions.
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

#Maintenant, on crée une fonction qui simule le changement d'état des parcelles de manière stochastique en fonction
#de la matrice de transition. La population au temps t+1 est déterminée de manière aléatoire, soit en utilisant 
#rand, en considérant les valeurs données par la matrice de transition comme des probabilités et non comme une distribution.
#Ici,chaque parcelle sera changée de manière indépendante en fonction de la probabilité qu'elle a de passer aux autres états. 
"""
_sim_stochastic
simule le changement d'état des parcelles de manière stochastique
change selon la matrice de transition
timeseries correspond à une matrice contenant les états pour chaque génération
transitions correspond à une matrice de probabilités de changement d'état
generation correspond à la génération actuelle, représentée par un nombre entier allant de 0 à n générations
"""
function _sim_stochastic!(timeseries, transitions, generation)
    for state in axes(timeseries, 1)
        pop_change = rand(Multinomial(timeseries[state, generation], transitions[state, :]))
        timeseries[:, generation+1] .+= pop_change
    end
end

#Ici, l'effectif sera modifié de manière déterministe, c'est à dire que les valeurs contenues dans la matrice de transition
#seront utilisées comme des distributions et seront appliquées directement à la population. Par exemple, si une l'état 1 a
#50% de chances de passer à l'état 2, 50% de l'effectif de l'état 1 sera modifié à l'état 2. Il n'y a donc une seule
#solution possible à chaque génération.
"""
_sim_determ
change les états des parcelles de manière déterministe, donc calcule le nombre de parcelle pour
chaque état selon les probabilités, mais pas de manière aléatoire.
timeseries correspond à une matrice contenant les états pour chaque génération
transitions correspond à une matrice de probabilités de changement d'état
generation correspond à la génération actuelle, représentée par un nombre entier allant de 0 à n générations
"""
function _sim_determ!(timeseries, transitions, generation)
    pop_change = (timeseries[:, generation]' * transitions)'
    timeseries[:, generation+1] .= pop_change
end

#Maintenant, on rassemble toutes les fonctions en une pour effecture la simulation. On s'assure d'abord que les conditions
#de taille de la matrice, du vecteur et du contenu de la matrice énoncées plus haut sont respectées. La simulation peut soit 
#être stochastique soit déterministe. À la fin, on arrondit les valeurs obtenues à la baisse, puisque comme ce qui est représenté
#est des parcelles, elles ne peuvent être représentées que par des nombres entiers. 
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

    return floor.(timeseries)
end

# ##
# States : vecteur qui contient les effectifs de chaque état
# Barren, Grass, Shrub1, Shrub2
s = [150, 15, 20, 25]
states = length(s)
patches = sum(s)

# Transitions : matrice qui définit les probabilités des transition de chaque état à un autre état
T = zeros(Float64, states, states)
T[1,:] = [0.96, 0.03, 0.005, 0.005]
T[2,:] = [0.10, 0.85, 0.03, 0.02]
T[3,:] = [0.05, 0.06, 0.89, 0.0]
T[4,:] = [0.05, 0.06, 0.0, 0.89]
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

# pour savoir le pourcentage de simulations qui respectent les critères, on effectue 100 simulations 
# stochastiques et on compte le nombre de simulations qui respectent les critères. 

simvalide = 0

for _ in 1:100
    global simvalide
    sim = simulation(T, s; stochastic=true, generations=200)
    final=floor.(sim[:,end]) #on arrondit à la baisse les valeurs d'états 
    végétalisé = (final[2]+final[3]+final[4])
    grass = (final[2])
    shrubs = (final[3]+final[4])
    shrubmin = (min(final[3], final[4]))
    if végétalisé<=40
        if grass ==(0.30 * végétalisé)
                if shrubmin >= (0.30*shrubs)
                simvalide += 1
            end
            
        end
    end
end
for _ in 1:100
    sim = simulation(T, s; stochastic=false, generations=200)
    println(final)
end
println(simvalide)

# # Présentation des résultats

# La figure suivante représente l’évolution du nombre de parcelles dans chaque état  
# (barren, grass, shrub 1 et shrub 2) au cours de 200 générations. Les lignes pâles correspondent aux différentes 
# simulations stochastiques, tandis que les lignes pleines et plus épaisses représentent la simulation déterministe basée 
# sur la matrice de transition. 

# On observe que le système converge vers un état d’équilibre après environ quelques dizaines de générations. À l’équilibre, 
# le nombre de parcelles barren se stabilise autour de 160, tandis que les parcelles occupées par les herbes (grass) 
# atteignent environ 10. Les deux espèces de buissons (shrub1 et shrub2) se stabilisent respectivement autour de 17 et 
# 10 parcelles.

# Ainsi, le nombre total de parcelles végétalisées à l’équilibre est d’environ 37, ce qui respecte la contrainte maximale 
# de 40 parcelles végétalisées. La proportion d’herbes (10 parcelles) est légèrement inférieure à la valeur cible de 
# 12 parcelles, tandis que le nombre total de buissons (27 parcelles) est proche de la cible de 28 parcelles. De plus, la 
# contrainte de diversité est respectée, puisque l’espèce de buisson la moins abondante représente 10 parcelles, soit plus 
# que le minimum requis de 9.

# Les simulations stochastiques montrent une certaine variabilité autour de ces valeurs, mais la tendance générale demeure 
# stable et cohérente avec la simulation déterministe. Toutefois, seulement 24 % des simulations respectent l’ensemble des 
# contraintes, ce qui est inférieur au seuil requis de 80 %.


# # Discussion

# Le choix de la population initiale repose sur la contrainte selon laquelle un maximum de 50 parcelles 
# peuvent être plantées. Une plus grande proportion de buissons a été introduite pour s’assurer qu’ils soient 
#bien représentés à l’équilibre, tandis que les herbes ont été introduites en plus faible quantité, en supposant 
#qu’elles pourraient coloniser les parcelles vides au fil du temps.

# Les résultats obtenus montrent que la matrice de transition choisie permet d’atteindre un état 
# d’équilibre relativement stable qui respecte plusieurs des contraintes imposées. En effet, le nombre 
# total de parcelles végétalisées demeure inférieur au maximum de 40 parcelles, et la répartition entre 
# les buissons respecte la contrainte de diversité minimale.

# Toutefois, le taux de simulations respectant l’ensemble des contraintes est de 24 %, ce qui est bien en 
# dessous du seuil requis de 80 %. On peut penser que cela est en grande partie dû au caractère stochastique 
# du modèle. Comme les transitions reposent sur du hasard, il devient difficile d’obtenir aussi souvent des 
# proportions aussi précises que celles demandées (par exemple, exactement autour de 30 % d’herbes). Prenant cela 
#en considération, les contraintes imposées sont assez restrictives, et il est possible que des critères légèrement 
# plus flexibles (par exemple, des intervalles plus larges, entre autres) auraient permis d’obtenir un taux de réussite 
# plus élevé.

# Finalement, ces résultats montrent que même si le modèle permet d’atteindre en moyenne, et une fois à l'équilibre,
# des valeurs proches de celles attendues, il reste sensible aux fluctuations aléatoires.

