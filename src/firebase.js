import { initializeApp } from "firebase/app";
import { getDatabase, ref, update } from "firebase/database";
import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  query,
  orderBy,
  limit,
  serverTimestamp,
} from "firebase/firestore";

const firebaseConfig = {
  apiKey:            "AIzaSyAr4IYnykpwovqOJWzfBd7abVdAma_Ig3Q",
  authDomain:        "diet-planner-3bdf3.firebaseapp.com",
  databaseURL:       "https://diet-planner-3bdf3-default-rtdb.firebaseio.com",
  projectId:         "diet-planner-3bdf3",
  storageBucket:     "diet-planner-3bdf3.firebasestorage.app",
  messagingSenderId: "927878354911",
  appId:             "1:927878354911:web:2e616b171a267b9910566a",
  measurementId:     "G-MSYWCM58MT",
};

const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);
const realtimeDb = getDatabase(firebaseApp);

function asNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

export async function saveScanResult(data) {
  const nutrition = data?.nutrition || {};
  const realtimePayload = {
    FoodName: data.food_name || "",
    Weight: asNumber(data.weight),
    Confidence: asNumber(data.confidence),
    Calories: asNumber(nutrition.calories),
    Protein: asNumber(nutrition.protein),
    Carbohydrates: asNumber(nutrition.carbohydrates),
    Fat: asNumber(nutrition.fat),
    Fiber: asNumber(nutrition.fiber),
    Sugar: asNumber(nutrition.sugar),
    CalorieClass: data.calorie_class || "",
    RecognitionModel: data.recognition_model || "",
    UpdatedAt: new Date().toISOString(),
  };

  await update(ref(realtimeDb, "Fruit_Analy"), realtimePayload);

  const docRef = await addDoc(collection(db, "nutrition_scans"), {
    food_name:  data.food_name,
    confidence: data.confidence,
    weight:     data.weight,
    nutrition:  data.nutrition,
    calorie_class: data.calorie_class || "",
    recognition_model: data.recognition_model || "",
    timestamp:  serverTimestamp(),
  });
  return docRef.id;
}

export async function getScanHistory(count = 12) {
  const q = query(
    collection(db, "nutrition_scans"),
    orderBy("timestamp", "desc"),
    limit(count)
  );
  const snap = await getDocs(q);
  return snap.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
}
