test_prompts = [
        # Wonder Woman
        "Show me Wonder Woman",
        "Create a detailed image of Wonder Woman wielding her Lasso of Truth in a city setting",
        "An intricate scene featuring Wonder Woman in her iconic costume, deflecting bullets with her bracelets amidst a World War II battlefield",

        # Shrek
        "Show me Shrek",
        "Create a detailed image of Shrek and Donkey in a swamp landscape",
        "An intricate scene featuring Shrek, Fiona, and Donkey on a quest through a fairy tale forest, encountering various magical creatures",

        # Elsa
        "Show me Elsa",
        "Create a detailed image of Elsa creating an ice palace in the mountains",
        "An intricate scene featuring Elsa using her ice powers to protect Arendelle from a magical threat, surrounded by intricate snowflake patterns",

        # Buzz Lightyear
        "Show me Buzz Lightyear",
        "Create a detailed image of Buzz Lightyear flying through a toy-filled bedroom",
        "An intricate scene featuring Buzz Lightyear leading a team of toys on a rescue mission in a complex, multi-level playset environment",

        # Spiderman
        "Show me Spiderman",
        "Create a detailed image of Spiderman swinging between New York City skyscrapers",
        "An intricate scene featuring Spiderman battling multiple villains across a detailed cityscape, showcasing his agility and web-slinging abilities",

        # Mario
        "Show me Mario",
        "Create a detailed image of Mario racing through a colorful Mushroom Kingdom level",
        "An intricate scene featuring Mario and Luigi navigating a complex, multi-world adventure with various power-ups and enemies from the franchise",

        # Pikachu
        "Show me Pikachu",
        "Create a detailed image of Pikachu using Thunderbolt in a Pokémon battle arena",
        "An intricate scene featuring Pikachu leading a group of diverse Pokémon through a challenging forest filled with obstacles and hidden Pokéballs",

        # Iron Man
        "Show me Iron Man",
        "Create a detailed image of Iron Man flying through a futuristic cityscape",
        "An intricate scene featuring Iron Man in his workshop, surrounded by holographic displays and various suits, working on a new nanotechnology armor",

        # Batman
        "Show me Batman",
        "Create a detailed image of Batman perched on a gargoyle overlooking Gotham City",
        "An intricate scene featuring Batman infiltrating a high-tech villain's lair, using various gadgets from his utility belt to overcome security systems",

        # Minions
        "Show me Minions",
        "Create a detailed image of Minions causing chaos in a banana factory",
        "An intricate scene featuring Minions building an elaborate Rube Goldberg machine to accomplish a simple task, with various mishaps occurring",
        
        # Elon Musk
        "Portrait of Elon Musk",
        "Elon Musk presenting a new Tesla car on stage",
        "Elon Musk standing on Mars next to a SpaceX rocket, Earth visible in the sky",

        # Keanu Reeves
        "Headshot of Keanu Reeves",
        "Keanu Reeves in John Wick costume, holding two pistols",
        "Keanu Reeves as Neo dodging bullets in slow-motion, surrounded by green Matrix code",

        # Beyonce
        "Close-up of Beyonce's face",
        "Beyonce performing on stage in a glittering costume",
        "Beyonce as an Egyptian queen, sitting on a golden throne surrounded by pyramids",

        # Chris Hemsworth
        "Photo of Chris Hemsworth smiling",
        "Chris Hemsworth as Thor, wielding Mjolnir with lightning in the background",
        "Chris Hemsworth surfing a giant wave while holding his Thor hammer",

        # Meryl Streep
        "Portrait of Meryl Streep",
        "Meryl Streep accepting an Oscar on stage",
        "Meryl Streep in costumes from her most famous roles, arranged in a group photo",

        # Emma Stone
        "Headshot of Emma Stone with red hair",
        "Emma Stone dancing in a yellow dress on a Los Angeles rooftop",
        "Emma Stone as Cruella de Vil, surrounded by dalmatians and fashion designs",

        # Dwayne Johnson
        "Portrait of Dwayne 'The Rock' Johnson",
        "Dwayne Johnson in the jungle, wearing adventure gear",
        "Dwayne Johnson as a Greek god, sitting on Mount Olympus with lightning bolts",

        # Taylor Swift
        "Close-up of Taylor Swift's face",
        "Taylor Swift performing on stage with a guitar",
        "Taylor Swift in different outfits representing her album eras, arranged in a circle",

        # Leonardo DiCaprio
        "Headshot of Leonardo DiCaprio",
        "Leonardo DiCaprio on a sinking ship, reaching out dramatically",
        "Leonardo DiCaprio in costumes from his most famous roles, walking on a red carpet",

        # Tesla
        "Tesla Model S in white",
        "Tesla Cybertruck driving through a futuristic city",
        "Flying Tesla cars in a sci-fi cityscape with renewable energy sources visible",

        # Starbucks
        "Starbucks coffee cup with logo",
        "Interior of a busy Starbucks cafe with baristas and customers",
        "Giant Starbucks cup as a building, with people entering through the lid",

        # Nike
        "Nike sneaker with swoosh logo",
        "Athlete wearing full Nike gear sprinting on a track",
        "Futuristic Nike store with holographic shoes and AI assistants",

        # McDonald's
        "McDonald's Big Mac burger",
        "McDonald's restaurant exterior at night with golden arches glowing",
        "Fantastical McDonald's theme park with rides shaped like menu items",

        # Coca-Cola
        "Classic glass Coca-Cola bottle",
        "Polar bear drinking Coca-Cola in a snowy landscape",
        "Coca-Cola waterfall pouring into a giant glass in a lush, tropical setting",

        # Apple
        "Apple iPhone on a plain background",
        "Apple Store interior with products on display and customers",
        "Futuristic city where all technology is Apple-branded, including flying cars",

        # LEGO
        "Single red LEGO brick",
        "Child building a colorful LEGO castle",
        "Life-sized city made entirely of LEGO bricks with minifigure citizens",

        # BMW
        "BMW logo on a car grille",
        "BMW M3 sports car racing on a mountain road",
        "Concept BMW flying car hovering over a futuristic cityscape",

        # The Starry Night
        "Van Gogh's The Starry Night painting",
        "Museum gallery with The Starry Night as the central piece",
        "Real night sky transforming into Van Gogh's The Starry Night style",

        # The Last Supper
        "Da Vinci's The Last Supper painting",
        "Restoration artist working on The Last Supper fresco",
        "3D recreation of The Last Supper scene with lifelike figures",

        # Mona Lisa
        "Close-up of Mona Lisa's face",
        "Mona Lisa hanging in the Louvre with tourists taking photos",
        "Multiple versions of Mona Lisa in different art styles side by side",

        # Creation of Adam
        "Michelangelo's Creation of Adam fresco",
        "Sistine Chapel ceiling with Creation of Adam highlighted",
        "3D sculpture of the hands from Creation of Adam in a modern art gallery",

        # The Raft of the Medusa
        "Painting of The Raft of the Medusa",
        "Stormy sea with a crowded raft of shipwreck survivors, inspired by Géricault",
        "Dramatic scene of desperate castaways on a makeshift raft, with a distant ship on the horizon, in the style of Romantic art",

        # Girl with a Pearl Earring
        "Portrait of Girl with a Pearl Earring",
        "Young woman in a blue and yellow turban with a large pearl earring, looking over her shoulder",
        "Vermeer-style painting of a girl with luminous skin, wearing an exotic headdress and a single pearl earring, against a dark background"
    ]
