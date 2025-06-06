const textos = [
  `Olá! Eu sou o Max!<br><br>
  Caso não queira ouvir minha voz, clique no botão 🔊 acima para me deixar mudo.
  <br><br>Estou aqui para te falar sobre Potencial Elétrico, Linhas Equipotenciais e Linhas de Campo.
  <br><br>A animação mostrará como as linhas de campo elétrico e equipotenciais se comportam no espaço ao redor dessas cargas enquanto elas se movem.
  <br><br> ...<br>
  `,

  `
  Ao aumentar o valor de uma carga (por exemplo: de 1 nC para 5 nC):
  <br>- Aumenta a intensidade do campo elétrico próximo.
  <br>- As linhas de campo ficam mais densas.
  <br>- O potencial elétrico se intensifica (mais cor).<br><br> ...<br>`,

  `Se você tornar uma carga negativa:
  <br>- Inverte o sentido das linhas de campo (entram na carga).
  <br>- A carga fica azul no gráfico.
  <br><br>Se usar cargas com sinais opostos (dipolo):
  <br>- Campo mais forte entre elas.
  <br>- Equipotenciais se curvam fortemente.
  <br><br>Com cargas do mesmo sinal:
  <br>- Linhas se repelem.
  <br>- Campo fraco entre elas.
  <br>- Equipotenciais mais espaçadas.
    <br><br>Insira os valores das cargas na parte superior da página e clique no botão Simular.`
];

let indiceTexto = 0;

function atualizarTexto() {
  const p = document.getElementById("texto-maxwell");
  p.innerHTML = textos[indiceTexto];

  // Controle de visibilidade das setas
  const btnAnterior = document.getElementById("anterior");
  const btnProximo = document.getElementById("proximo");

  if (btnAnterior) btnAnterior.style.display = (indiceTexto === 0) ? "none" : "inline-block";
  if (btnProximo) btnProximo.style.display = (indiceTexto === textos.length - 1) ? "none" : "inline-block";
}


function navegarTexto(direcao) {
  indiceTexto += direcao;
  if (indiceTexto < 0) indiceTexto = 0;
  if (indiceTexto >= textos.length) indiceTexto = textos.length - 1;
  atualizarTexto();
}

// Inicializa o primeiro texto no carregamento
window.addEventListener("load", () => {
  atualizarTexto();
});


let leituraAutomatica = true;
function lerTextoPersonalizado(texto) {
  if (!leituraAutomatica) return;

  const textoElement = document.querySelector("#fala-maxwell p");
  const imagemMaxwell = document.querySelector(".maxwell");

  if (!textoElement || !imagemMaxwell) {
    console.error("Texto do balão ou imagem do Maxwell não encontrado!");
    return;
  }

  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }

  const sintese = new SpeechSynthesisUtterance(texto);
  sintese.lang = "pt-BR";

  const vozes = window.speechSynthesis.getVoices();
  const vozPtBr = vozes.find(voz => voz.lang === 'pt-BR');
  if (vozPtBr) sintese.voice = vozPtBr;

  let bocaAberta = true;
  const bocaAbertaSrc = "/static/assets/Maxwell_Infantil.png";
  const bocaFechadaSrc = "/static/assets/Maxwell_Infantil_bocaFechada.png";

  const animacaoFala = setInterval(() => {
    if (!window.speechSynthesis.speaking) {
      clearInterval(animacaoFala);
      imagemMaxwell.src = bocaAbertaSrc;
      return;
    }

    imagemMaxwell.src = bocaAberta ? bocaFechadaSrc : bocaAbertaSrc;
    bocaAberta = !bocaAberta;
  }, 500);

  sintese.onend = () => {
    clearInterval(animacaoFala);
    imagemMaxwell.src = bocaAbertaSrc;
  };

  window.speechSynthesis.speak(sintese);
}

function lerTexto() {
  if (!leituraAutomatica) return;

  const texto = textos[indiceTexto];
  lerTextoPersonalizado(texto);
}

function toggleAudio() {
  const botao = document.getElementById("botaoAudio");

  if (leituraAutomatica) {
    // Desativar leitura
    window.speechSynthesis.cancel();
    leituraAutomatica = false;
    botao.textContent = "🔇"; // Mostra que está no modo silencioso
    const imagemMaxwell = document.querySelector(".maxwell");
    if (imagemMaxwell) {
      imagemMaxwell.src = "/static/assets/Maxwell_Infantil.png";
    }
  } else {
    // Ativar leitura novamente
    leituraAutomatica = true;
    botao.textContent = "🔊"; // Mostra que está no modo com som
    lerTexto();
  }
}

window.addEventListener("load", () => {
  atualizarTexto(); // Atualiza apenas o texto
  if (leituraAutomatica) {
    lerTexto();     // Faz a leitura somente uma vez no carregamento
  }
});



document.getElementById("atualizarSimulacao").addEventListener("click", atualizarSimulacao);
let textosSimulacao = [];
let indiceTextoSimulacao = 0;

function atualizarTextoSimulacao() {
    const p = document.getElementById("texto-maxwell");
    p.innerHTML = textosSimulacao[indiceTextoSimulacao];

    // Botões de navegação
    const btnAnterior = document.getElementById("anterior");
    const btnProximo = document.getElementById("proximo");

    if (btnAnterior) btnAnterior.style.display = (indiceTextoSimulacao === 0) ? "none" : "inline-block";
    if (btnProximo) btnProximo.style.display = (indiceTextoSimulacao === textosSimulacao.length - 1) ? "none" : "inline-block";
    
    lerTextoPersonalizado(textosSimulacao[indiceTextoSimulacao]);
}

function navegarTextoSimulacao(direcao) {
    indiceTextoSimulacao += direcao;
    if (indiceTextoSimulacao < 0) indiceTextoSimulacao = 0;
    if (indiceTextoSimulacao >= textosSimulacao.length) indiceTextoSimulacao = textosSimulacao.length - 1;
    atualizarTextoSimulacao();
}
function atualizarSimulacao() {
    const carga1 = parseFloat(document.getElementById("valor_carga1").value);
    const carga2 = parseFloat(document.getElementById("valor_carga2").value);
    const img = document.getElementById("simulacaoGIF");
    const loader = document.getElementById("loader");
    const timestamp = new Date().getTime();

    // Verifica sinais das cargas
    if (carga1 * carga2 < 0) {// Cargas com sinais opostos (dipolo)
        textoSimulacao = `
        <p>Essa simulação representa o campo elétrico e o potencial elétrico gerados por duas cargas puntiformes em movimento com sinais opostos (uma carga positiva e uma negativa).
        <br><br> As linhas de campo saem da carga positiva e entram na carga negativa. 
        <br><br> Entre as cargas, essas linhas se adensam, indicando um campo elétrico mais forte nessa região. <br><br> ...<br></p>
        <p>Elas se curvam de uma carga para a outra, mostrando a atração entre as cargas.
       <br><br>  As linhas equipotenciais representam regiões onde o potencial elétrico é constante. 
       <br><br>  No caso de um dipolo, essas linhas se curvam acentuadamente ao redor das cargas e entre elas. 
      <br> <br>  Próximo às cargas, onde o campo é mais intenso, as equipotenciais estão mais próximas umas das outras. A diferença de potencial entre as cargas é significativa.</p>

        `;
    } else if (carga1 * carga2 > 0) {// Cargas com o mesmo sinal
        textoSimulacao = `  
         <p> Essa simulação representa o campo elétrico e o potencial elétrico gerados por duas cargas puntiformes em movimento com o mesmo sinal (ambas positivas ou ambas negativas).

        <br> <br>  As linhas de campo saem de ambas as cargas (se forem positivas) ou entram em ambas as cargas (se forem negativas) e se repelem mutuamente.
       <br><br>  Entre as cargas, essas linhas se afastam umas das outras e se tornam menos densas (rarefazem), indicando um campo elétrico mais fraco nessa região central.
        <br> <br>  Elas se curvam para longe da região central entre as cargas, mostrando a repulsão entre elas.<br><br> ...<br></p>

         <p> As linhas equipotenciais representam regiões onde o potencial elétrico é constante.
        <br><br>   No caso de cargas de mesmo sinal, essas linhas tendem a envolver cada carga individualmente e, na região entre as cargas, elas se tornam mais espaçadas e se abrem, refletindo o afastamento das linhas de campo.
         <br><br>  Próximo a cada carga individual, onde o campo é mais intenso devido àquela carga, as equipotenciais estão mais próximas. 
         <br><br> No entanto, na região entre as duas cargas, onde o campo resultante é mais fraco, as equipotenciais ficam visivelmente mais espaçadas.
         <br> <br> Isso indica que a variação do potencial elétrico é menos acentuada na região entre as cargas.</p>
        `; 
    }

    // Atualiza o texto do balão
    textosSimulacao = textoSimulacao.split("</p>").filter(p => p.trim()).map(p => p + "</p>");
    indiceTextoSimulacao = 0;
    atualizarTextoSimulacao();

    // Lê o texto, se estiver ativado
    lerTextoPersonalizado(textoSimulacao);

    // Mostrar loader e esconder GIF enquanto carrega
    loader.style.display = "block";
    img.style.display = "none";

    fetch(`/gerar_gif?valor_carga1=${carga1}&valor_carga2=${carga2}&_=${timestamp}`)
        .then(response => {
            if (!response.ok) throw new Error("Erro ao gerar o GIF.");

            img.onload = () => {
                loader.style.display = "none";
                img.style.display = "block";
            };

            img.src = `/static/simulacao.gif?_=${timestamp}`;
        })
        .catch(error => {
            loader.style.display = "none";
            console.error("Erro na geração do GIF:", error);
        });
}
function handleNavegacao(direcao) {
    if (textosSimulacao.length > 0) {
        navegarTextoSimulacao(direcao);
    } else {
        navegarTexto(direcao);
    }
}

