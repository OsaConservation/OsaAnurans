const species_path = "../specieslist.csv"
const container = document.querySelector(".card-container")
const scoreEl = document.getElementById("score")

window.addEventListener('load', function() {
    loadScore();
})

container.addEventListener("click", (e)=>{
    if(e.target.matches("button")){
        const name = e.target.value;
        const answer = e.target.parentNode.dataset.value
        if(name === answer){
            console.log("hooray")
            score += 10
        }else{
            score -= 10
        }
        updateScore();
        loadScore()

        container.replaceChildren()
        parseFile();

    }
} )

function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                // const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                const data = []
                const all_names = []

                // Loop through each row and extract the 'scientific_name' column
                results.data.forEach(row => {
                    if(row.display === '1'){
                        data.push(row)
                        all_names.push(row.scientific_name)
                    }    
                });

                let index = Math.floor(Math.random() * all_names.length)
                row = results.data[index]
                all_names.splice(index, 1)

                paths = generateQuestion(index, data)
                audio_path = paths[0]
                spec_path = paths[1]

                const audio = new Audio(audio_path);
                audio.controls = true;
                const spec = document.createElement("img")
                spec.src = spec_path;
                spec.classList.add("question-spec")

                const QuestionCard = document.createElement("div")
                QuestionCard.classList.add("question-card")



                container.appendChild(QuestionCard);
                QuestionCard.appendChild(spec)
                QuestionCard.appendChild(audio)

                answers = generateAnswers(row.scientific_name, all_names)

                const AnswerCard = document.createElement("div")
                AnswerCard.classList.add("answer-card")
                AnswerCard.dataset.value = row.scientific_name
                container.appendChild(AnswerCard)

                for(let i = 0; i < answers.length; i++){
                    let button = document.createElement("button")
                    button.textContent = answers[i]
                    button.value = answers[i]
                    AnswerCard.appendChild(button)
                }

                // filter data where display ==1
                // randomly select species
                // randomly select image/audio 
                // get species name + three random species for answers
                // display all 
                // function to check if clicked answer is correct
                //     Display response / frog picture?
                // generate new question 
                //     Track # correct?
            }
        })
}

function generateQuestion(index, data){
    row = data[index]

    
    audio_files = Papa.parse(row.audio_files).data[0]
    spec_files = Papa.parse(row.spec_files).data[0]

    specIndex = Math.floor(Math.random() * audio_files.length)


    let audio_path = "static/" + row.scientific_name + "/audio/" + audio_files[specIndex]
    let spec_path = "static/" + row.scientific_name + "/img/spec/" + spec_files[specIndex]



    return [audio_path, spec_path]

}

function generateAnswers(scientific_name, all_names){

    answers = [scientific_name]

    for(let i = 0; i < 3; i++){
        let randomIndex = Math.floor(Math.random() * all_names.length)
        answers.push(all_names[randomIndex])

        all_names.splice(randomIndex, 1)
    }

    return shuffle(answers)



}

function checkAnswer(answer){

}

function shuffle(array) {
  var m = array.length, t, i;

  // While there remain elements to shuffle…
  while (m) {

    // Pick a remaining element…
    i = Math.floor(Math.random() * m--);

    // And swap it with the current element.
    t = array[m];
    array[m] = array[i];
    array[i] = t;
  }

  return array;
}

let score = JSON.parse(localStorage.getItem("score"))

function updateScore(){
    localStorage.setItem("score", JSON.stringify(score))
    scoreEl.innerText = score;

}

function clearScore(){
    score = 0;
    updateScore();
}

function loadScore(){
    let score = JSON.parse(localStorage.getItem("score"));
    if(!score){
        score = 0;
    }

    scoreEl.innerText = score
}
parseFile();