package com.parkingauthorityticketingsystem;

import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.AutoCompleteTextView;
import android.widget.EditText;

import com.facebook.stetho.Stetho;
import com.parkingauthorityticketingsystem.location_related.LocationService;

public class MainActivity extends AppCompatActivity implements LocationFragment.InterfaceTextClicked, RequestLocInterface {

    String communication;
    static int justOnce = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Stetho.initializeWithDefaults(this);
        LocationFragment ff = new LocationFragment();

        FragmentManager fm = getFragmentManager();
        FragmentTransaction ft = fm.beginTransaction();
        ft.add(R.id.fragmentContainer,new CreateShiftlogFragment(),"CreateShiftlog").commit();

        getFragmentManager().beginTransaction().add(R.id.fragmentContainer,new LocationFragment(),"LocationFrag").commit();

    }

    @Override
    public void sendText(String text){
       if(text.contains("coordinates")){
          //  System.out.println("Coordinates being sent");
           CreateShiftlogFragment frag = (CreateShiftlogFragment)
                   getFragmentManager().findFragmentByTag("CreateShiftlog");
            frag.getCoordinates(text);
        }else if(text.contains("map")){
           SL_ListViewFragment lvfrag = (SL_ListViewFragment)
                   getFragmentManager().findFragmentByTag("ListViewFrag");
           lvfrag.getCoordinates(text);
       }
    }

    @Override
    public void requestLocation(String a){
        LocationFragment frag = (LocationFragment)
                getFragmentManager().findFragmentByTag("LocationFrag");
        if(a.equals("saveLoc")){      //coordinates needed to save into DB
            frag.saveloc(0);
        }else if(a.equals("saveLoc1")){
            frag.saveloc(1);
        }else{
            System.out.println("Argument not saveLoc");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        System.out.println("Inside Main's onActivityResult");

        View v = Viewholder.getMyView();

//        ((EditText)v.findViewById(R.id.max_time_et)).setText("55");


        if (requestCode == LocationFragment.LOCATION_REQUEST_CODE) {

            LocationFragment frag = (LocationFragment)
                    getFragmentManager().findFragmentByTag("LocationFrag");

            String ans =  data.getExtras().getString(LocationService.LOC_RESULT_EXTRA);
            System.out.println("From oAR: "+ans);

            switch(resultCode){
                case LocationService.INITIAL_CODE :  String[] result = ans.split(",");
                                                        ((AutoCompleteTextView)v.findViewById(R.id.street_name)).setText(result[0]);
                                                    break;
                case LocationService.STREETS_CODE : frag.updateText(ans);
                                                    break;
            }
        }


    }

    public static class Viewholder{
        static View myView;

        static void setMyView(View view){
            myView = view;
        }
        static View getMyView(){
            return myView;
        }
    }



}
