## ChatGPA

**ChatGPA** est une IA super intelligente qui dispose d'une multitude de
fonctionalités, telles que:
 - Prédire qui a envoyé un message
 - C'est tout
 - Nan vraiment y'a rien d'autre
 - Mais c'est un bot Discord donc c'est cool

## Installation

### NixOS

```bash
git clone https://github.com/PoustouFlan/ChatGPA.git
cd ChatGPA
```
Ensuite, modifier le fichier `configuration.yaml`, qui doit
ressembler à :
```yaml
token:      "LeTokenDe.Votre.Bot_Ici"
guild_id:   123456789012345678
```
en remplaçant la valeur de `guild_id` par l'identifiant de votre serveur.

Enfin, pour lancer le bot, vous pouvez exécuter :
```bash
nix-shell --run make
```

### Pas NixOS

Débrouillez-vous ptdr, allez bon courage
