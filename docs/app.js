const species_path = "specieslist.csv"

const container = document.querySelector(".card-container")
const popup_card = document.getElementById("popup-card");
const popup_image = document.getElementById("popup-image");
const popup_close = document.getElementById("popup-close");
const popup_close_image = document.getElementById("popup-close-image");
const backdrop = document.getElementById('backdrop');


container.addEventListener("click", (e)=>{
    if(e.target.matches("button.popup-button")){
        const name = e.target.value;

        Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true,
            complete: function(results){

                
                for(let i=0; i < results.data.length; i++){
                    if(results.data[i].scientific_name != name){
                        continue
                    }else{
                        row = results.data[i];
                        common_name = row.common_name
                        code = row.code
                        endemism = row.Endemism
                        family_name = row.family_name
                        // genus_name = row.genus_name
                        // species_name = row.species_name
                        range = row.range
                        habitat = row.habitat
                        threats = row.threats
                        IUCN_status = row.IUCN_status
                        IUCN_population_trend = row.IUCN_population_trend
                        IUCN_assessement_year = row.IUCN_assessement_year
                        audio_files = Papa.parse(row.audio_files).data[0]
                        spec_files = Papa.parse(row.spec_files).data[0]
                        createPopupCard(name, common_name, code, endemism, family_name, range, habitat, threats, IUCN_status, IUCN_population_trend, IUCN_assessement_year, audio_files, spec_files)
                    }
                }
                
            }
        })
    }else if(e.target.matches("img.card-spec")){
        createImagePopup(e.target.src)
    }
})

popup_close.addEventListener("click", ()=> {
    popup_card.replaceChildren(popup_card.firstElementChild)
    popup_image.replaceChildren(popup_image.firstElementChild)
    popup_image.style.display = "none"
    popup_card.style.display = "none"
    backdrop.style.display = "none"
})

popup_close_image.addEventListener("click", ()=> {
    popup_card.replaceChildren(popup_card.firstElementChild)
    popup_image.replaceChildren(popup_image.firstElementChild)
    popup_image.style.display = "none"
    popup_card.style.display = "none"
    backdrop.style.display = "none"
})


backdrop.addEventListener("click", ()=>{
    popup_image.replaceChildren(popup_image.firstElementChild)
    popup_card.replaceChildren(popup_card.firstElementChild)
    popup_image.style.display = "none"
    popup_card.style.display = "none"
    backdrop.style.display = "none"
})




function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                // const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                const listElement = document.getElementById("card-container");

                // Loop through each row and extract the 'scientific_name' column
                results.data.forEach(row => {
                    if(row.display === '1'){
                        let newCard = createFrogCard(row.common_name, row.scientific_name, row.code);
                        
                        listElement.appendChild(newCard);
                    }    
                });
            }
        })
}


function createFrogCard(common_name, scientific_name, code){
    const newCard = document.createElement("div");
    newCard.classList.add("frog-card")
    const scientific = document.createElement("i");
    scientific.textContent = scientific_name;

    const common = document.createElement("p");
    common.textContent = common_name;

    const codeEl = document.createElement("p")
    codeEl.textContent = code;
    

    let Audiopath = "docs/static/" + scientific_name + "/audio/" + scientific_name + "0.mp3";
    let Imagepath = "docs/static/"+ scientific_name + "/img/"  + scientific_name + "0.webp";
    let Specpath = "docs/static/"+ scientific_name + "/img/spec/"  + scientific_name + "0.webp";

    // let Audiopath = "docs/audio/" + scientific_name + ".mp3";
    // let Imagepath = "docs/img/" + scientific_name + ".webp";
    // let Specpath = "docs/img/spec/" + scientific_name + ".webp";


    const audio = new Audio(Audiopath);
    // audio.onerror = function(){
    //     this.parentNode.style.display='none';};
    audio.controls = true;

    const img = document.createElement("img");
    img.src = Imagepath

    const spec = document.createElement("img")
    spec.src = Specpath;
    spec.classList.add("card-spec")

    const button = document.createElement("button");
    button.innerHTML = "See more"
    button.classList.add("popup-button");
    button.value = scientific_name

    
    

    newCard.appendChild(img);
    newCard.appendChild(spec);
    newCard.appendChild(scientific);
    newCard.appendChild(common);
    newCard.appendChild(codeEl);
    newCard.appendChild(audio);
    newCard.appendChild(button);

    return newCard

}

function createPopupCard(scientific_name, common_name, code, endemism, family_name, range, habitat, threats, IUCN_status, IUCN_population_trend, IUCN_assessement_year, audio_files, spec_files){


    const popup_container = document.createElement("div")
    popup_container.id = "popup-container"


    const title = document.createElement("h1");
    title.innerHTML = scientific_name.italics();
    const common = document.createElement("p");
    common.innerHTML = "Common name(s): " + common_name



    let Imagepath = "docs/img/" + scientific_name + ".webp";
    const img = document.createElement("img");
    img.src = Imagepath
    img.id = "frog-picture"

    console.log(audio_files)
    console.log(spec_files)

    var examples = audio_files.map((e, i) => [e, spec_files[i]])
    console.log(examples)
    examples.forEach((file) =>{

        console.log(file)

        let Audiopath = "docs/static/" + scientific_name + "/audio/" + file[0]
        let audio = new Audio(Audiopath)
        audio.controls = true
        

        let Specpath = "docs/static/" + scientific_name + "/img/spec/" + file[1]

        let spec = document.createElement("img")
        spec.src = Specpath;
        spec.classList.add("spec")
        spec.classList.add("spec-card")

        let record = document.createElement("div")
        record.appendChild(spec)
        record.appendChild(audio)
        popup_container.appendChild(record);

    })


    const codeEl = document.createElement("p")
    codeEl.textContent = "Annotation Code: " + code;

    const endemismEl = document.createElement("p")
    endemismEl.textContent = "Endemism: " + endemism;

    const family_nameEl = document.createElement("p")
    family_nameEl.textContent = "Family: " + family_name

    const rangeEl = document.createElement("p")
    rangeEl.textContent = "Range: " + range

    const habitatEl = document.createElement("p")
    habitatEl.textContent = "Habitat: " + habitat

    const threatsEl = document.createElement("p")
    threatsEl.textContent = "Threats: " + threats

    const IUCN_statusEl = document.createElement("p")
    IUCN_statusEl.textContent = "Conservation Status: " + IUCN_status

    const IUCN_population_trendEl = document.createElement("p")
    IUCN_population_trendEl.textContent = "Population Trend: " + IUCN_population_trend

    const IUCN_assessement_yearEl = document.createElement("p")
    IUCN_assessement_yearEl.textContent = "Last Assessed: " + IUCN_assessement_year






    

    popup_card.appendChild(img);
    popup_card.appendChild(title)
    popup_card.appendChild(common)
    popup_card.appendChild(codeEl)
    popup_card.appendChild(endemismEl)
    popup_card.appendChild(family_nameEl)
    popup_card.appendChild(rangeEl)
    popup_card.appendChild(habitatEl)
    popup_card.appendChild(threatsEl)
    popup_card.appendChild(IUCN_statusEl)
    popup_card.appendChild(IUCN_population_trendEl)
    popup_card.appendChild(IUCN_assessement_yearEl)



    
    popup_card.appendChild(popup_container)


    // popup_container.appendChild(spec);
    // popup_container.appendChild(audio);
    popup_card.style.display = "block"
    backdrop.style.display = "block"

}

function createImagePopup(source){

    const img = new Image();
    img.src = source
    img.classList.add("popup-image")

    popup_image.appendChild(img)
    popup_image.style.display = "block"
    backdrop.style.display = "block"

}



parseFile();
