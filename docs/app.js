const species_path = "specieslist.csv"

const container = document.querySelector(".card-container")
const popup_card = document.getElementById("popup-card");
const popup_close = document.getElementById("popup-close");
const backdrop = document.getElementById('backdrop');


container.addEventListener("click", (e)=>{
    if(e.target.matches("button.popup-button")){
        const name = e.target.value;

        Papa.parse(species_path, {
            download: true,
            header: true,
            complete: function(results){

                
                for(let i=0; i < results.data.length; i++){
                    if(results.data[i].scientific_name != name){
                        continue
                    }else{
                        row = results.data[i];
                        common_name = row.common_name
                        createPopupCard(name, common_name)
                    }
                }
                
            }
        })
    }
})

popup_close.addEventListener("click", ()=> {
    popup_card.style.display = "none"
    backdrop.style.display = "none"
})

backdrop.addEventListener("click", ()=>{
    popup_card.style.display = "none"
    backdrop.style.display = "none"
})




function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                const listElement = document.getElementById("card-container");

                // Loop through each row and extract the 'scientific_name' column
                data.forEach(row => {
                    if(row.display === '1'){
                        let newCard = createFrogCard(row.common_name, row.scientific_name);
                        
                        listElement.appendChild(newCard);
                    }    
                });
            }
        })
}


function createFrogCard(common_name, scientific_name){
    const newCard = document.createElement("div");
    newCard.classList.add("frog-card")
    const scientific = document.createElement("i");
    scientific.textContent = scientific_name;

    const common = document.createElement("p");
    common.textContent = common_name;
    

    let Audiopath = "docs/audio/" + scientific_name + ".mp3";
    let Imagepath = "docs/img/" + scientific_name + ".webp";
    let Specpath = "docs/img/spec/" + scientific_name + ".webp";
    const audio = new Audio(Audiopath);
    // audio.onerror = function(){
    //     this.parentNode.style.display='none';};
    audio.controls = true;

    const img = document.createElement("img");
    img.src = Imagepath

    const spec = document.createElement("img")
    spec.src = Specpath;

    const button = document.createElement("button");
    button.innerHTML = "See more"
    button.classList.add("popup-button");
    button.value = scientific_name

    
    

    newCard.appendChild(img);
    newCard.appendChild(spec);
    newCard.appendChild(scientific);
    newCard.appendChild(common);
    newCard.appendChild(audio);
    newCard.appendChild(button);

    return newCard

}


function createPopupCard(scientific_name, common_name){




    popup_card.replaceChildren(popup_card.firstElementChild)


    const title = document.createElement("h1");
    title.innerHTML = scientific_name.italics();
    const common = document.createElement("p");
    common.innerHTML = common_name


    let Audiopath = "docs/audio/" + scientific_name + ".mp3";
    let Imagepath = "docs/img/" + scientific_name + ".webp";
    let Specpath = "docs/img/spec/" + scientific_name + ".webp";
    const audio = new Audio(Audiopath);
    audio.controls = true;

    const img = document.createElement("img");
    img.src = Imagepath
    img.id = "frog-picture"

    const spec = document.createElement("img")
    spec.src = Specpath;
    spec.classList.add("spec")



    const popup_container = document.createElement("div")
    popup_container.id = "popup-container"


    popup_card.appendChild(title)
    popup_card.appendChild(common)
    popup_card.appendChild(img);

    
    popup_card.appendChild(popup_container)


    popup_container.appendChild(spec);
    popup_container.appendChild(audio);
    popup_card.style.display = "block"
    backdrop.style.display = "block"

}




parseFile();
